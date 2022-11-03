// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
/// @brief Represents an element of the input tensor and contains additional information about such element needed
///        by the implementation of the Unique operation
template <typename Index_t>
struct Element {
    Element(Index_t idx_) : idx{idx_} {}
    Element(Index_t idx_, Index_t rev_idx_, int64_t count_) : idx{idx_}, rev_idx{rev_idx_}, count{count_} {}
    Index_t idx = 0;       // the index of the current element in the original input tensor
    Index_t rev_idx = -1;  // the index of the unique element in the output tensor
    int64_t count = 1;     // the number of occurrences of the current element in the original input tensor
};

template <typename Index_t>
struct UniqueElements {
    std::vector<Element<Index_t>> all_tensor_elements;
    std::vector<Element<Index_t>> unique_tensor_elements;
};

namespace {
template <typename T>
std::vector<Element<T>> generate_data_references(const size_t count) {
    std::vector<Element<T>> data_references;
    data_references.reserve(count);

    for (T i = 0; i < count; ++i) {
        data_references.emplace_back(i);
    }
    return data_references;
}
}  // namespace

template <typename Data_t, typename Index_t>
UniqueElements<Index_t> find_unique_elements(const Data_t* data,
                                             const Shape& data_shape,
                                             const std::unique_ptr<int64_t> axis,
                                             const bool sorted) {
    using std::begin;
    using std::end;

    const auto elements_are_equal = [&data](const Element<Index_t>& lhs, const Element<Index_t>& rhs) {
        return *(data + lhs.idx) == *(data + rhs.idx);
    };

    const auto already_unique = [&elements_are_equal](const Element<Index_t>& existing_unique_elem) {
        return [&elements_are_equal, &existing_unique_elem](const Element<Index_t>& x) {
            return elements_are_equal(existing_unique_elem, x);
        };
    };

    const auto data_elems_count = shape_size(data_shape);
    UniqueElements<Index_t> ret;
    ret.all_tensor_elements = generate_data_references<Index_t>(data_elems_count);
    ret.all_tensor_elements[0].rev_idx = 0;
    ret.unique_tensor_elements.push_back(ret.all_tensor_elements[0]);

    if (data_shape.size() == 0 || data_shape.size() == 1 && data_shape[0] == 1) {
        // NTD
    } else if (data_shape.size() == 1 && data_shape[0] > 1) {
        for (size_t i = 1; i < data_elems_count; ++i) {
            auto& data_elem_descriptor = ret.all_tensor_elements[i];
            auto existing_unique = std::find_if(begin(ret.unique_tensor_elements),
                                                end(ret.unique_tensor_elements),
                                                already_unique(data_elem_descriptor));
            if (existing_unique != end(ret.unique_tensor_elements)) {
                data_elem_descriptor.rev_idx = existing_unique->rev_idx;
                existing_unique->count++;
            } else {
                data_elem_descriptor.rev_idx = ret.unique_tensor_elements.size();
                ret.unique_tensor_elements.push_back(data_elem_descriptor);
            }
        }
    } else {
        throw std::runtime_error("Not implemented yet");
    }

    return ret;
}

template <typename Data_t, typename Index_t>
void unique(Data_t* out_unique_elements,
            Index_t* out_indices,
            Index_t* out_rev_indices,
            int64_t* out_counts,
            const Data_t* data,
            // const Shape& data_shape,
            const UniqueElements<Index_t>& unique_elements) {
    for (size_t i = 0; i < unique_elements.unique_tensor_elements.size(); ++i) {
        const auto& descriptor = unique_elements.unique_tensor_elements[i];
        out_unique_elements[i] = *(data + descriptor.idx);
        out_indices[i] = descriptor.idx;
        out_counts[i] = descriptor.count;
    }

    for (size_t i = 0; i < unique_elements.all_tensor_elements.size(); ++i) {
        const auto& descriptor = unique_elements.all_tensor_elements[i];
        out_rev_indices[i] = descriptor.rev_idx;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
