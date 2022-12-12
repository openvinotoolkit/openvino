// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
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

    const auto& strides_shape = input_shapes.size() < 4 ? op->get_input_shape(3) : input_shapes[3];
    NODE_VALIDATION_CHECK(op,
                          strides_shape.rank().compatible(1),
                          "Strides input must be 1D (strides rank: ",
                          strides_shape.rank(),
                          ").");

    // it is not possible to define output shape if input data shape rank is undefined
    // even the lengths of begin, end, or strides are defined
    if (input_shape.rank().is_dynamic()) {
        output_shapes[0] = ov::PartialShape::dynamic();
        return;
    }
    auto input_rank = input_shape.size();

    const auto get_input_bounds = [&](size_t idx) {
        std::vector<int64_t> lower, upper;
        if (!get_data_as_int64<T>(idx, op, lower, constant_data)) {
            // if no const data try get input bounds
            auto bounds = ngraph::evaluate_both_bounds(op->get_input_source_output(idx));

            if (bounds.first && bounds.second) {
                lower = std::make_shared<op::v0::Constant>(bounds.first)->cast_vector<int64_t>();
                upper = std::make_shared<op::v0::Constant>(bounds.second)->cast_vector<int64_t>();
            }
        }

        return std::make_pair(lower, upper);
    };

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
    auto begin = get_input_bounds(1);
    auto end = get_input_bounds(2);
    auto got_begin = !begin.first.empty();
    auto got_end = !end.first.empty();

    std::vector<int64_t> strides;
    bool got_strides = false;

    if (input_shapes.size() > 3) {
        got_strides = get_data_as_int64<T>(3, op, strides, constant_data);
    } else if (got_begin) {
        // generate default strides
        strides.resize(begin.first.size(), 1);
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
        AxisSet axis_set;
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
            for (size_t i = axis + 1; i < begin.first.size(); ++i) {
                if (new_axis_mask.count(i)) {
                    num_new_axis_after_ellipses++;
                }
            }

            int64_t num_input_axis_after_ellipses =
                (number_axes - axis - num_new_axis_after_ellipses - 1);  // -1 because it's a position of ellipses
            int64_t num_of_hidden_dims = input_rank - num_input_axis_after_ellipses - num_input_axis_before_ellipses;
            for (int64_t i = 0; i < num_of_hidden_dims; ++i, ++input_shape_idx) {
                dims.emplace_back(input_shape[input_shape_idx]);
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
                // set default value for stride or use given value
                auto stride = (strides.size() > static_cast<size_t>(axis)) ? strides[axis] : static_cast<int64_t>(1);
                NODE_VALIDATION_CHECK(op, stride != 0, "Stride must be non-zero");
                // normalize by add max to value if negative
                const auto normalize = [](const int64_t& value, const int64_t& max) -> int64_t {
                    return (value < 0) ? value + max : value;
                };

                // clip value to min, max
                const auto clip = [](const int64_t& value, const int64_t& min, const int64_t& max) -> int64_t {
                    return std::min(std::max(value, min), max);
                };

                // get stride output dimension for dimension and bounds
                // may not be called for stride 0 (div by 0!!!) assert check done above
                const auto get_output_dim = [&](const int64_t& dim, const int64_t& lower, const int64_t& upper) {
                    const auto is_reverse_stride = stride < 0;

                    constexpr int64_t lower_min = 0;
                    const int64_t lower_max = is_reverse_stride ? dim - 1 : dim;
                    const int64_t upper_min = is_reverse_stride ? -1 : lower_min;
                    const int64_t default_min = is_reverse_stride ? lower_max : lower_min;
                    const int64_t default_max = is_reverse_stride ? -1 : dim;

                    auto lb = begin_mask.count(axis) ? default_min : clip(normalize(lower, dim), lower_min, lower_max);
                    auto ub = end_mask.count(axis) ? default_max : clip(normalize(upper, dim), upper_min, dim);

                    // decrees range by modifing lower bound depends on stride direction
                    is_reverse_stride ? --lb : ++lb;

                    if ((is_reverse_stride && lb >= ub) || (!is_reverse_stride && lb <= ub)) {
                        return ((ub - lb) / stride) + 1;
                    } else {
                        return static_cast<int64_t>(0);
                    }
                };

                const auto& begin_lb = begin.first[axis];
                const auto& begin_ub = begin.second.empty() ? begin_lb : begin.second[axis];

                const auto& end_lb = end.first[axis];
                const auto& end_ub = end.second.empty() ? end_lb : end.second[axis];

                if (input_shape[input_shape_idx].is_dynamic()) {
                    // the relationship between input and output length is monotonically increasing
                    // so we repeat the dimension inference twice to infer dynamic dimension
                    const auto& interval = input_shape[input_shape_idx].get_interval();
                    auto lb = get_output_dim(interval.get_min_val(), begin_ub, end_lb);
                    auto ub =
                        interval.has_upper_bound() ? get_output_dim(interval.get_max_val(), begin_lb, end_ub) : -1;
                    dims.emplace_back(lb, ub);
                } else {
                    const auto& dimension = input_shape[input_shape_idx].get_length();
                    auto lb = get_output_dim(dimension, begin_ub, end_lb);
                    auto ub = get_output_dim(dimension, begin_lb, end_ub);
                    dims.emplace_back(lb, ub);
                }

                if (std::is_same<DimType, ov::Dimension>::value && dims.back() == input_shape[input_shape_idx]) {
                    // for equal ov::Dimension do merge to get input label (always success)
                    DimType::merge(dims.back(), dims.back(), input_shape[input_shape_idx]);
                }

                input_shape_idx++;
            } else {
                if (input_shape[input_shape_idx].is_static()) {
                    auto dim_value = input_shape[input_shape_idx].get_length();
                    dims.emplace_back(0, dim_value);
                } else {
                    dims.emplace_back(-1);
                }

                input_shape_idx++;
            }
        }
    }

    // get remaining values
    for (; input_shape_idx < input_shape.rank().get_length(); ++input_shape_idx) {
        dims.push_back(input_shape[input_shape_idx]);
    }

    output_shapes[0] = T(std::move(dims));
}
}  // namespace v1
}  // namespace op
}  // namespace ov
