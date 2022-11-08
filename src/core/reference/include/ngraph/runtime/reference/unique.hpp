// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
/// @brief Represents an element of the input tensor
template <typename Index_t, typename Count_t>
struct Element {
    Element(Index_t idx_) : idx{idx_} {}
    /// The index of the current element in the original input tensor. It never changes even if the elements get sorted.
    /// This value is used as a mapping between a unique element in the first output tensor and the position of this
    /// element in the original input tensor.
    Index_t idx = 0;
    /// The rev_idx is a mapping between every element in the original input and the location of a unique element
    /// in the first output tensor. More than one Element can have the same rev_idx.
    Index_t rev_idx = -1;
    /// The number of occurrences of a given element in the input tensor. This value is different than one only for
    /// duplicates found in the input tensor.
    Count_t count = 1;  // the number of occurrences of the current element in the original input tensor
};

template <typename Index_t, typename Counts_t>
struct UniqueElements {
    std::vector<Element<Index_t, Counts_t>> all_tensor_elements;
    std::vector<Element<Index_t, Counts_t>> unique_tensor_elements;
};

namespace {
template <typename T, typename C>
std::vector<Element<T, C>> generate_element_descriptors(const size_t count) {
    std::vector<Element<T, C>> descriptors;
    descriptors.reserve(count);

    for (T i = 0; i < count; ++i) {
        descriptors.emplace_back(i);
    }
    return descriptors;
}

bool scalar_or_single_element(const Shape& s) {
    return s.size() == 0 || std::all_of(std::begin(s), std::end(s), [](Shape::value_type d) {
               return d == 1;
           });
}
}  // namespace

template <typename Data_t, typename Index_t, typename Counts_t = int64_t>
UniqueElements<Index_t, Counts_t> find_unique_elements(const Data_t* data,
                                                       const Shape& data_shape,
                                                       std::unique_ptr<int64_t> axis,
                                                       const bool sorted) {
    using std::begin;
    using std::end;

    const auto ascending_order = [&data](const Element<Index_t, Counts_t>& lhs, const Element<Index_t, Counts_t>& rhs) {
        return *(data + lhs.idx) < *(data + rhs.idx);
    };

    const auto elements_are_equal = [&data](const Element<Index_t, Counts_t>& lhs,
                                            const Element<Index_t, Counts_t>& rhs) {
        return *(data + lhs.idx) == *(data + rhs.idx);
    };

    const auto already_unique = [&elements_are_equal](const Element<Index_t, Counts_t>& existing_unique_elem) {
        return [&elements_are_equal, &existing_unique_elem](const Element<Index_t, Counts_t>& x) {
            return elements_are_equal(existing_unique_elem, x);
        };
    };

    const auto data_elems_count = shape_size(data_shape);
    UniqueElements<Index_t, Counts_t> ret;
    ret.all_tensor_elements = generate_element_descriptors<Index_t, Counts_t>(data_elems_count);

    if (sorted) {
        std::sort(begin(ret.all_tensor_elements), end(ret.all_tensor_elements), ascending_order);
    }

    if (scalar_or_single_element(data_shape)) {
        ret.all_tensor_elements[0].rev_idx = 0;
        ret.unique_tensor_elements.push_back(ret.all_tensor_elements[0]);
    } else if (!axis || (data_shape.size() == 1 && data_shape[0] > 1)) {
        ret.all_tensor_elements[0].rev_idx = 0;
        ret.unique_tensor_elements.push_back(ret.all_tensor_elements[0]);
        for (size_t i = 1; i < data_elems_count; ++i) {
            auto& tensor_element = ret.all_tensor_elements[i];
            auto existing_unique = end(ret.unique_tensor_elements);
            if (sorted) {
                existing_unique = std::lower_bound(begin(ret.unique_tensor_elements),
                                                   end(ret.unique_tensor_elements),
                                                   tensor_element,
                                                   ascending_order);
            } else {
                existing_unique = std::find_if(begin(ret.unique_tensor_elements),
                                               end(ret.unique_tensor_elements),
                                               already_unique(tensor_element));
            }

            if (existing_unique != end(ret.unique_tensor_elements)) {
                tensor_element.rev_idx = existing_unique->rev_idx;
                existing_unique->count++;
            } else {
                tensor_element.rev_idx = ret.unique_tensor_elements.size();
                ret.unique_tensor_elements.push_back(tensor_element);
            }
        }
    } else {
        throw std::runtime_error("Not implemented yet");
    }

    return ret;
}

template <typename Index_t, typename Counts_t = int64_t>
std::tuple<Shape, Shape, Shape> make_tensor_shapes(const UniqueElements<Index_t, Counts_t>& unique_elements) {
    const auto output0 = Shape{unique_elements.unique_tensor_elements.size()};
    const auto output1_3 = output0;  // TODO
    const auto output2 = Shape{unique_elements.all_tensor_elements.size()};
    return std::make_tuple(output0, output1_3, output2);
}

template <typename Data_t, typename Index_t, typename Counts_t = int64_t>
void unique(Data_t* out_unique_elements,
            Index_t* out_indices,
            Index_t* out_rev_indices,
            Counts_t* out_counts,
            const Data_t* data,
            const UniqueElements<Index_t, Counts_t>& unique_elements) {
    for (size_t i = 0; i < unique_elements.unique_tensor_elements.size(); ++i) {
        const auto& descriptor = unique_elements.unique_tensor_elements[i];
        out_unique_elements[i] = *(data + descriptor.idx);
        out_indices[i] = descriptor.idx;
        out_counts[i] = descriptor.count;
    }

    // filling out this output tensor requires a separate pass over all elements of the input tensor
    // for each input element we need to output and index fo that element in the first output tensor
    // additionally if sorting was involved the "all_tensor_elements" might be ordered differently than the elements
    // in the original input tensor - this is why descriptor.idx is used for indexing the output tensor below
    for (const auto& descriptor : unique_elements.all_tensor_elements) {
        out_rev_indices[descriptor.idx] = descriptor.rev_idx;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
