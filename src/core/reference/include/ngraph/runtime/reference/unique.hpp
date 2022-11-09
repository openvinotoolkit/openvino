// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename Index_t, typename Count_t>
struct TensorSlice {
    TensorSlice() = delete;
    TensorSlice(const Index_t idx_) : idx{idx_}, single_value{true} {}
    TensorSlice(const Index_t idx_, const Index_t rev_idx_, const Count_t count_)
        : idx{idx_},
          rev_idx{rev_idx_},
          count{count_},
          single_value{true} {}
    TensorSlice(const Index_t idx_, const Shape& first_elem_coord_, const size_t slice_elems_)
        : idx{idx_},
          first_elem_coord{first_elem_coord_},
          slice_elems{slice_elems_} {}
    /// The index of the current element in the original input tensor. It never changes even if the elements get sorted.
    /// This value is used as a mapping between a unique element in the first output tensor and the position of this
    /// element in the original input tensor.
    Index_t idx = 0;
    /// The rev_idx is a mapping between every element in the original input and the location of a unique element
    /// in the first output tensor. More than one Element can have the same rev_idx.
    Index_t rev_idx = -1;
    /// The number of occurrences of a given element in the input tensor. This value is different than one only for
    /// duplicates found in the input tensor.
    Count_t count = 1;
    Shape first_elem_coord;
    size_t slice_elems;
    /// Indicates if this object points to a single value in the input tensor (rather than a slice of the tensor)
    bool single_value = false;
};

template <typename Index_t, typename Count_t>
struct UniqueElements {
    std::vector<TensorSlice<Index_t, Count_t>> all_tensor_elements;
    std::vector<TensorSlice<Index_t, Count_t>> unique_tensor_elements;
};

namespace {
template <typename Index_t, typename Count_t>
std::vector<TensorSlice<Index_t, Count_t>> generate_single_value_descriptors(const size_t count) {
    std::vector<TensorSlice<Index_t, Count_t>> descriptors;
    descriptors.reserve(count);

    for (Index_t i = 0; i < count; ++i) {
        descriptors.emplace_back(i);
    }
    return descriptors;
}

template <typename Index_t, typename Count_t>
std::vector<TensorSlice<Index_t, Count_t>> generate_slice_descriptors(const Shape& data_shape, const int64_t axis) {
    std::vector<TensorSlice<Index_t, Count_t>> descriptors;
    descriptors.reserve(axis);

    auto shape_copy = data_shape;
    shape_copy.erase(shape_copy.begin() + axis);
    const auto tensor_slice_elems = shape_size(shape_copy);

    for (int64_t i = 0; i < data_shape[axis]; ++i) {
        // the coordinate of the first element in a given tensor slice
        auto first_elem_of_this_slice = Shape(data_shape.size(), 0);
        first_elem_of_this_slice[axis] = i;
        descriptors.emplace_back(i, first_elem_of_this_slice, tensor_slice_elems);
    }

    return descriptors;
}

bool scalar_or_single_element(const Shape& s) {
    return s.size() == 0 || std::all_of(std::begin(s), std::end(s), [](Shape::value_type d) {
               return d == 1;
           });
}
}  // namespace

template <typename Data_t, typename Index_t, typename Count_t = int64_t>
UniqueElements<Index_t, Count_t> find_unique_elements(const Data_t* data,
                                                      const Shape& data_shape,
                                                      std::unique_ptr<int64_t> axis,
                                                      const bool sorted) {
    using std::begin;
    using std::end;

    const auto ascending_order = [&data](const TensorSlice<Index_t, Count_t>& lhs,
                                         const TensorSlice<Index_t, Count_t>& rhs) {
        return *(data + lhs.idx) < *(data + rhs.idx);
    };

    const auto slices_are_equal = [&data](const TensorSlice<Index_t, Count_t>& lhs,
                                          const TensorSlice<Index_t, Count_t>& rhs) {
        return false;
    };

    const auto elements_are_equal = [&data](const TensorSlice<Index_t, Count_t>& lhs,
                                            const TensorSlice<Index_t, Count_t>& rhs) {
        return *(data + lhs.idx) == *(data + rhs.idx);
    };

    const auto already_unique = [&elements_are_equal](const TensorSlice<Index_t, Count_t>& existing_unique_elem) {
        return [&elements_are_equal, &existing_unique_elem](const TensorSlice<Index_t, Count_t>& x) {
            return elements_are_equal(existing_unique_elem, x);
        };
    };

    const auto already_unique_slice = [&slices_are_equal](const TensorSlice<Index_t, Count_t>& existing_unique_elem) {
        return [&slices_are_equal, &existing_unique_elem](const TensorSlice<Index_t, Count_t>& x) {
            return slices_are_equal(existing_unique_elem, x);
        };
    };

    UniqueElements<Index_t, Count_t> ret;

    if (scalar_or_single_element(data_shape)) {
        ret.all_tensor_elements.emplace_back(0, 0, 1);
        ret.unique_tensor_elements.emplace_back(0, 0, 1);
        return ret;
    } else if (!axis || (data_shape.size() == 1 && data_shape[0] > 1)) {  // 1D or N-D without any axis
        const auto data_elems_count = shape_size(data_shape);
        ret.all_tensor_elements = generate_single_value_descriptors<Index_t, Count_t>(data_elems_count);

        if (sorted) {
            std::sort(begin(ret.all_tensor_elements), end(ret.all_tensor_elements), ascending_order);
        }

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
        ret.all_tensor_elements = generate_slice_descriptors<Index_t, Count_t>(data_shape, *axis);

        ret.all_tensor_elements[0].rev_idx = 0;
        ret.unique_tensor_elements.push_back(ret.all_tensor_elements[0]);

        for (int64_t i = 1; i < data_shape[*axis]; ++i) {
            auto& tensor_element = ret.all_tensor_elements[i];
            auto existing_unique = end(ret.unique_tensor_elements);
            // if (sorted) {
            //     existing_unique = std::lower_bound(begin(ret.unique_tensor_elements),
            //                                        end(ret.unique_tensor_elements),
            //                                        tensor_element,
            //                                        ascending_order);
            // } else {
            existing_unique = std::find_if(begin(ret.unique_tensor_elements),
                                           end(ret.unique_tensor_elements),
                                           already_unique_slice(tensor_element));

            if (existing_unique != end(ret.unique_tensor_elements)) {
                tensor_element.rev_idx = existing_unique->rev_idx;
                existing_unique->count++;
            } else {
                tensor_element.rev_idx = ret.unique_tensor_elements.size();
                ret.unique_tensor_elements.push_back(tensor_element);
            }
        }
    }

    return ret;
}

template <typename Index_t, typename Count_t = int64_t>
std::tuple<Shape, Shape, Shape> make_tensor_shapes(const UniqueElements<Index_t, Count_t>& unique_elements,
                                                   const Shape& data_shape,
                                                   std::unique_ptr<int64_t> axis) {
    if (axis) {
        auto output0 = data_shape;
        output0[*axis] = unique_elements.unique_tensor_elements.size();
        const auto output1_3 = Shape{unique_elements.unique_tensor_elements.size()};
        const auto output2 = Shape{data_shape[*axis]};
        return std::make_tuple(output0, output1_3, output2);
    } else {
        const auto output0 = Shape{unique_elements.unique_tensor_elements.size()};
        const auto output1_3 = output0;
        const auto output2 = Shape{unique_elements.all_tensor_elements.size()};
        return std::make_tuple(output0, output1_3, output2);
    }
}

template <typename Data_t, typename Index_t, typename Count_t = int64_t>
void unique(Data_t* out_unique_elements,
            Index_t* out_indices,
            Index_t* out_rev_indices,
            Count_t* out_counts,
            const Data_t* data,
            const UniqueElements<Index_t, Count_t>& unique_elements) {
    for (size_t i = 0; i < unique_elements.unique_tensor_elements.size(); ++i) {
        const auto& descriptor = unique_elements.unique_tensor_elements[i];
        if (descriptor.single_value) {
            out_unique_elements[i] = *(data + descriptor.idx);
            out_indices[i] = descriptor.idx;
            out_counts[i] = descriptor.count;
        } else {
            std::cout << "Not implemented yet\n";
        }
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
