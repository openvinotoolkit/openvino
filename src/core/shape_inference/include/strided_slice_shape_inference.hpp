// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/strided_slice.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v1 {

template <class T>
void shape_infer(const StridedSlice* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4) && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto& begin_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op,
                          begin_shape.rank().compatible(1),
                          "Begin input must be 1D (begin rank: ",
                          begin_shape.rank(),
                          ").");

    const auto& end_shape = input_shapes[2];
    NODE_VALIDATION_CHECK(op,
                          end_shape.rank().compatible(1),
                          "End input must be 1D (end rank: ",
                          end_shape.rank(),
                          ").");

    const auto& strides_shape = input_shapes[3];
    NODE_VALIDATION_CHECK(op,
                          strides_shape.rank().compatible(1),
                          "End input must be 1D (end rank: ",
                          strides_shape.rank(),
                          ").");

    // it is not possible to define output shape if input data shape rank is undefined
    // even the lengths of begin, end, or strides are defined
    if (input_shape.rank().is_dynamic()) {
        output_shapes[0] = ov::PartialShape::dynamic();
        return;
    }
    auto input_rank = input_shape.size();

    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;

    auto number_elements_in_1d = [](const StridedSlice* op, const T& shape_1d) -> int64_t {
        auto rank_1d = shape_1d.rank();
        if (rank_1d.is_static()) {
            NODE_VALIDATION_CHECK(op, rank_1d.get_length() == 1, "Only 1D tensor is allowed.");
            if (shape_1d[0].is_static()) {
                return static_cast<int64_t>(shape_1d[0].get_length());
            }
        }
        return -1;
    };

    // compute constant values of begin, end, and strides if possible
    bool got_begin = get_data_as_int64<T>(1, op, begin, constant_data);
    bool got_end = get_data_as_int64<T>(2, op, end, constant_data);
    bool got_strides = false;
    if (input_shapes.size() > 3) {
        got_strides = get_data_as_int64<T>(3, op, strides, constant_data);
    } else if (got_begin) {
        // generate default strides
        strides.resize(begin.size(), 1);
        got_strides = true;
    }

    // compute and check a number of axes for which begin, end, and strides are defined
    auto number_axes = number_elements_in_1d(op, begin_shape);
    auto end_number_axes = number_elements_in_1d(op, end_shape);
    if (number_axes != -1 && end_number_axes != -1) {
        NODE_VALIDATION_CHECK(op, number_axes == end_number_axes, "Begin and end need to have same number of values.");
    } else if (end_number_axes != -1) {
        number_axes = end_number_axes;
    }
    auto strides_number_axes = number_elements_in_1d(op, strides_shape);
    if (number_axes != -1 && strides_number_axes != -1) {
        NODE_VALIDATION_CHECK(op,
                              number_axes == strides_number_axes,
                              "Stride needs to have same number of values as begin and end.");
    } else if (strides_number_axes != -1) {
        number_axes = strides_number_axes;
    }

    // if number of axes is undefined we cannot say about output rank
    if (number_axes < 0) {
        output_shapes[0] = ov::PartialShape::dynamic();
        return;
    }

    // collect indices of axes by which the shape needs to be changed
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        AxisSet axis_set{};
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                axis_set.emplace(i);
            }
        }
        return axis_set;
    };
    AxisSet ellipsis_mask = convert_mask_to_axis_set(op->get_ellipsis_mask());
    NODE_VALIDATION_CHECK(op, ellipsis_mask.size() <= 1, "At most one ellipsis is allowed.");
    AxisSet new_axis_mask = convert_mask_to_axis_set(op->get_new_axis_mask());
    AxisSet begin_mask = convert_mask_to_axis_set(op->get_begin_mask());
    AxisSet end_mask = convert_mask_to_axis_set(op->get_end_mask());
    AxisSet shrink_axis_mask = convert_mask_to_axis_set(op->get_shrink_axis_mask());
    NODE_VALIDATION_CHECK(op,
                          input_rank + new_axis_mask.size() >= static_cast<size_t>(number_axes),
                          "Input rank plus number of new axis has to be at least the size of Lower "
                          "and Upper bounds vector.");

    std::vector<DimType> dims;
    int64_t input_shape_idx = 0;
    for (int64_t axis = 0; axis < number_axes; ++axis) {
        // add all dimensions hidden under the ellipsis mask if ellipsis mask is set
        if (ellipsis_mask.count(axis)) {
            // only one bit in ellipsis mask is allowed
            int num_new_axis_after_ellipses = 0;
            int num_input_axis_before_ellipses = 0;
            for (int64_t i = 0; i < axis; ++i) {
                if (!new_axis_mask.count(i)) {
                    num_input_axis_before_ellipses++;
                }
            }
            for (size_t i = axis + 1; i < begin.size(); ++i) {
                if (new_axis_mask.count(i)) {
                    num_new_axis_after_ellipses++;
                }
            }

            int64_t num_input_axis_after_ellipses =
                (number_axes - axis - num_new_axis_after_ellipses - 1);  // -1 because it's a position of ellipses
            int64_t num_of_hidden_dims = input_rank - num_input_axis_after_ellipses - num_input_axis_before_ellipses;
            for (int64_t i = 0; i < num_of_hidden_dims; ++i) {
                dims.emplace_back(input_shape[input_shape_idx]);
                input_shape_idx++;
            }
        } else {
            // add new single dimension if new_axis_mask is set
            if (new_axis_mask.count(axis)) {
                dims.emplace_back(1);
            }
            // skip this dimension if shrink_axis_mask is set
            else if (shrink_axis_mask.count(axis)) {
                input_shape_idx++;
            }
            // calculating dimension (begin, end, begin_mask, end_mask, stride)
            else if (got_begin && got_end && got_strides) {
                const int64_t lb0 = begin[axis];
                const int64_t ub0 = end[axis];
                // set default value for stride or use given value
                int64_t stride = 1;
                if (strides.size() > static_cast<size_t>(axis)) {
                    stride = strides[axis];
                }
                NODE_VALIDATION_CHECK(op, stride != 0, "Stride must be non-zero");

                auto get_output_dim = [&](int64_t input_dim) {
                    // make a mutable copy
                    auto lb = lb0;
                    auto ub = ub0;
                    // convert negative indexes to positive
                    // take max for this case: if abs(lb) > input_shape[input_shape_idx],then after
                    // conversion lb < 0
                    // so according to tensorflow and numpy we just get 0
                    if (lb < 0) {
                        lb = std::max(input_dim + lb, int64_t(0));
                    }

                    if (ub < 0) {
                        ub = std::max(input_dim + ub, stride > 0 ? int64_t(0) : int64_t(-1));
                    }

                    // apply restrictions when begin or end values more than max possible values.
                    lb = std::min(input_dim, lb);
                    ub = std::min(input_dim, ub);

                    int64_t dimension = 0;
                    if (stride < 0) {
                        // apply masks
                        if (begin_mask.count(axis)) {
                            lb = input_dim - 1;
                        }
                        if (end_mask.count(axis)) {
                            ub = -1;
                        }

                        lb = std::min(lb, input_dim - 1);
                        lb -= 1;  // we always get 1st element, so we need decrease range
                        if (ub <= lb) {
                            dimension = (ub - lb) / stride + 1;
                        }
                    } else if (stride != 0) {
                        // apply masks
                        if (begin_mask.count(axis)) {
                            lb = 0;
                        }
                        if (end_mask.count(axis)) {
                            ub = input_dim;
                        }

                        lb += 1;  // we always get 1st element, so we need decrease range
                        if (ub >= lb) {
                            dimension = (ub - lb) / stride + 1;
                        }
                    }
                    return dimension;
                };

                if (input_shape[input_shape_idx].is_dynamic()) {
                    // the relationship between input and output length is monotonically increasing
                    // so we repeat the dimension inference twice to infer dynamic dimension
                    const Interval& interval = input_shape[input_shape_idx].get_interval();
                    int64_t odim_min = get_output_dim(interval.get_min_val());
                    int64_t odim_max;
                    if (interval.has_upper_bound())
                        odim_max = get_output_dim(interval.get_max_val());
                    else
                        odim_max = -1;

                    dims.emplace_back(ov::Dimension(odim_min, odim_max));
                } else {
                    int64_t dimension = get_output_dim(input_shape[input_shape_idx].get_length());
                    dims.emplace_back(dimension);
                }

                input_shape_idx++;
            } else {
                if (input_shape[input_shape_idx].is_static()) {
                    auto dim_value = input_shape[input_shape_idx].get_length();
                    dims.emplace_back(DimType(0, dim_value));
                } else {
                    dims.emplace_back(DimType(-1));
                }

                input_shape_idx++;
            }
        }
    }

    // get remaining values
    for (; input_shape_idx < input_shape.rank().get_length(); ++input_shape_idx) {
        dims.emplace_back(input_shape[input_shape_idx]);
    }

    output_shapes[0] = T(dims);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
