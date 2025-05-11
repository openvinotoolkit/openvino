// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gather.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {

enum class DescriptorType { SINGLE_VALUE, SLICE };

template <typename Index_t, typename Count_t>
struct TensorSlice {
    TensorSlice(const Index_t idx_, const DescriptorType descriptor_type_)
        : idx{idx_},
          descriptor_type{descriptor_type_} {}
    TensorSlice(const Index_t idx_, const Index_t rev_idx_, const Count_t count_)
        : idx{idx_},
          rev_idx{rev_idx_},
          count{count_} {}
    /// The index of the current element in the original input tensor. It never changes even if the elements get
    /// sorted. This value is used as a mapping between a unique element in the first output tensor and the position
    /// of this element in the original input tensor.
    Index_t idx = 0;
    /// The rev_idx is a mapping between every element in the original input and the location of a unique element
    /// in the first output tensor. More than one Element can have the same rev_idx.
    Index_t rev_idx = -1;
    /// The number of occurrences of a given element in the input tensor. This value is different than one only for
    /// duplicates found in the input tensor.
    Count_t count = 1;
    /// Indicates if this object points to a single value in the input tensor (rather than a slice of the tensor)
    DescriptorType descriptor_type = DescriptorType::SINGLE_VALUE;
};

template <typename Index_t, typename Count_t>
struct UniqueElements {
    /// Contains descriptors of all elements in the input tensor. Possibly sorted by value.
    std::vector<TensorSlice<Index_t, Count_t>> all_tensor_elements;
    /// Subset of all tensor elements. First occurrences of the unique values.
    std::vector<TensorSlice<Index_t, Count_t>> unique_tensor_elements;
    /// Axis (optional). Used to gather unique elements over a given dimension.
    int64_t axis = 0;
};

namespace {

// Generates descriptors of slices or individual elems of the input tensor. This function returns a vector of
// helper objects representing elements that the "unique" algorithm is supposed to process later.
template <typename Index_t, typename Count_t>
std::vector<TensorSlice<Index_t, Count_t>> generate_descriptors(const size_t count, const DescriptorType type) {
    std::vector<TensorSlice<Index_t, Count_t>> descriptors;
    descriptors.reserve(count);

    for (Index_t i = 0, end = static_cast<Index_t>(count); i < end; ++i) {
        descriptors.emplace_back(i, type);
    }

    return descriptors;
}

// Returns indices of the first element of each tensor slice. The index is equal to a coordinate index.
template <typename Index_t, typename Count_t>
inline std::pair<size_t, size_t> first_elems_of_both_slices(const TensorSlice<Index_t, Count_t>& lhs,
                                                            const TensorSlice<Index_t, Count_t>& rhs,
                                                            const std::vector<size_t>& data_shape_strides,
                                                            const int64_t axis) {
    return {data_shape_strides[axis] * lhs.idx, data_shape_strides[axis] * rhs.idx};
}

template <typename Index_t, typename Count_t>
inline size_t calc_slices_offset(const TensorSlice<Index_t, Count_t>& lhs,
                                 const TensorSlice<Index_t, Count_t>& rhs,
                                 const std::vector<size_t>& data_shape_strides,
                                 const int64_t axis) {
    const auto first_elem_indices = first_elems_of_both_slices(lhs, rhs, data_shape_strides, axis);
    if (first_elem_indices.first > first_elem_indices.second) {
        return first_elem_indices.first - first_elem_indices.second;
    } else {
        return first_elem_indices.second - first_elem_indices.first;
    }
}

inline Shape slice_shape_to_iterate(Shape data_shape, const int64_t axis) {
    data_shape.erase(data_shape.begin() + axis, data_shape.begin() + axis + 1);
    return data_shape;
}

bool scalar_or_single_element(const Shape& s) {
    return std::all_of(std::begin(s), std::end(s), [](Shape::value_type d) {
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

    const auto data_shape_strides = row_major_strides(data_shape);

    if (axis && *axis < 0) {
        const auto normalized_axis = *axis + data_shape.size();
        *axis = normalized_axis;
    }

    const auto ascending_order = [&data](const TensorSlice<Index_t, Count_t>& lhs,
                                         const TensorSlice<Index_t, Count_t>& rhs) {
        return *(data + lhs.idx) < *(data + rhs.idx);
    };

    int64_t axisVal = 0;
    if (axis) {
        axisVal = *axis;
        if (axisVal < 0) {
            axisVal += data_shape.size();
        }
    }

    const auto slices_ascending_order = [&](const TensorSlice<Index_t, Count_t>& lhs,
                                            const TensorSlice<Index_t, Count_t>& rhs) {
        const auto shape_to_iterate = slice_shape_to_iterate(data_shape, axisVal);

        for (auto it = CoordinateIterator(shape_to_iterate); it != CoordinateIterator::end(); ++it) {
            auto elem_coord_lhs = *it;
            elem_coord_lhs.insert(elem_coord_lhs.cbegin() + axisVal, lhs.idx);

            auto elem_coord_rhs = *it;
            elem_coord_rhs.insert(elem_coord_rhs.cbegin() + axisVal, rhs.idx);

            const auto lhs_elem_idx = coordinate_index(elem_coord_lhs, data_shape);
            const auto rhs_elem_idx = coordinate_index(elem_coord_rhs, data_shape);

            if (*(data + lhs_elem_idx) < *(data + rhs_elem_idx)) {
                return true;
            } else if (*(data + lhs_elem_idx) > *(data + rhs_elem_idx)) {
                return false;
            } else {
                continue;
            }
        }

        return false;
    };

    const auto elements_are_equal = [&data](const TensorSlice<Index_t, Count_t>& lhs,
                                            const TensorSlice<Index_t, Count_t>& rhs) {
        return *(data + lhs.idx) == *(data + rhs.idx);
    };

    const auto slices_are_equal = [&](const TensorSlice<Index_t, Count_t>& lhs,
                                      const TensorSlice<Index_t, Count_t>& rhs) {
        const auto& slice_with_lower_idx =
            std::min(lhs, rhs, [](const TensorSlice<Index_t, Count_t>& a, const TensorSlice<Index_t, Count_t>& b) {
                return a.idx < b.idx;
            });

        // the individual elements in the two compared slices are always separated by the same offset
        // and this can be used to compare them elementwise
        const auto slices_offset = calc_slices_offset(lhs, rhs, data_shape_strides, axisVal);
        const auto shape_to_iterate = slice_shape_to_iterate(data_shape, axisVal);

        for (auto it = CoordinateIterator(shape_to_iterate); it != CoordinateIterator::end(); ++it) {
            // All slice elements have a "slice index" constant value at the axis position, only the other dimensions
            // vary for each slice element. Those dimensions are provided by CoordinateIterator, the value at axis
            // needs to be injected manually.
            auto elem_coord = *it;
            elem_coord.insert(elem_coord.cbegin() + axisVal, slice_with_lower_idx.idx);
            const auto lhs_elem_idx = coordinate_index(elem_coord, data_shape);
            const auto rhs_elem_idx = lhs_elem_idx + slices_offset;
            if (*(data + lhs_elem_idx) != *(data + rhs_elem_idx)) {
                return false;
            }
        }
        return true;
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
    } else if (!axis || (is_vector(data_shape) && data_shape[0] > 1)) {  // 1D or N-D without any axis
        const auto data_elems_count = shape_size(data_shape);
        ret.all_tensor_elements =
            generate_descriptors<Index_t, Count_t>(data_elems_count, DescriptorType::SINGLE_VALUE);

        if (sorted) {
            std::stable_sort(begin(ret.all_tensor_elements), end(ret.all_tensor_elements), ascending_order);
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
                tensor_element.rev_idx = static_cast<Index_t>(ret.unique_tensor_elements.size());
                ret.unique_tensor_elements.push_back(tensor_element);
            }
        }
    } else {
        ret.axis = axisVal;
        ret.all_tensor_elements = generate_descriptors<Index_t, Count_t>(data_shape[axisVal], DescriptorType::SLICE);

        if (sorted) {
            std::stable_sort(begin(ret.all_tensor_elements), end(ret.all_tensor_elements), slices_ascending_order);
        }
        ret.all_tensor_elements[0].rev_idx = 0;
        ret.unique_tensor_elements.push_back(ret.all_tensor_elements[0]);

        for (size_t i = 1; i < data_shape[axisVal]; ++i) {
            auto& tensor_element = ret.all_tensor_elements[i];
            auto existing_unique = end(ret.unique_tensor_elements);

            if (sorted) {
                existing_unique = std::lower_bound(begin(ret.unique_tensor_elements),
                                                   end(ret.unique_tensor_elements),
                                                   tensor_element,
                                                   slices_ascending_order);
            } else {
                existing_unique = std::find_if(begin(ret.unique_tensor_elements),
                                               end(ret.unique_tensor_elements),
                                               already_unique_slice(tensor_element));
            }

            if (existing_unique != end(ret.unique_tensor_elements)) {
                tensor_element.rev_idx = existing_unique->rev_idx;
                existing_unique->count++;
            } else {
                tensor_element.rev_idx = static_cast<Index_t>(ret.unique_tensor_elements.size());
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
        if (*axis < 0) {
            const auto normalized_axis = *axis + data_shape.size();
            *axis = normalized_axis;
        }
        // if the axis was specified we need to return a data shape with a modified dimension-at-axis
        // this is where we need to insert the number of detected unique elements
        // all other dimensions stay the same as in the original data_shape
        int64_t axisVal = 0;
        if (axis) {
            axisVal = *axis;
            if (axisVal < 0) {
                axisVal += data_shape.size();
            }
        }
        auto output0 = data_shape;
        output0[axisVal] = unique_elements.unique_tensor_elements.size();
        const auto output1_3 = Shape{unique_elements.unique_tensor_elements.size()};
        const auto output2 = Shape{data_shape[axisVal]};
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
            const Shape& data_shape,
            const Shape& out_shape,
            const UniqueElements<Index_t, Count_t>& descriptors) {
    if (descriptors.unique_tensor_elements[0].descriptor_type == DescriptorType::SINGLE_VALUE) {
        for (size_t i = 0; i < descriptors.unique_tensor_elements.size(); ++i) {
            const auto& descriptor = descriptors.unique_tensor_elements[i];
            out_unique_elements[i] = *(data + descriptor.idx);
            out_indices[i] = descriptor.idx;
            out_counts[i] = descriptor.count;
        }
    } else {
        std::vector<Index_t> indices;
        indices.reserve(descriptors.unique_tensor_elements.size());

        for (size_t i = 0; i < descriptors.unique_tensor_elements.size(); ++i) {
            const auto& descriptor = descriptors.unique_tensor_elements[i];
            out_indices[i] = descriptor.idx;
            out_counts[i] = descriptor.count;

            indices.push_back(descriptor.idx);
        }

        gather(data,
               indices.data(),
               out_unique_elements,
               data_shape,
               Shape{descriptors.unique_tensor_elements.size()},
               out_shape,
               descriptors.axis);
    }

    // filling out this output tensor requires a separate pass over all elements of the input tensor
    // for each input element we need to output and index fo that element in the first output tensor
    // additionally if sorting was involved the "all_tensor_elements" might be ordered differently than the elements
    // in the original input tensor - this is why descriptor.idx is used for indexing the output tensor below
    for (const auto& descriptor : descriptors.all_tensor_elements) {
        out_rev_indices[descriptor.idx] = descriptor.rev_idx;
    }
}
}  // namespace reference
}  // namespace ov
