// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/range.hpp>

#include "utils.hpp"
namespace ov {
namespace op {

namespace ShapeInferRange {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> range_shape_infer(const Node* op,
                                       const std::vector<T>& input_shapes,
                                       bool output_is_integral,
                                       bool step_allows_zero,
                                       const ITensorAccessor& tensor_accessor) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3));

    NODE_VALIDATION_CHECK(op, input_shapes[0].rank().compatible(0), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[1].rank().compatible(0), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[2].rank().compatible(0), "'step' input is not a scalar");

    const auto start_val = get_input_const_data_as<TRShape, double>(op, 0, tensor_accessor);
    const auto stop_val = get_input_const_data_as<TRShape, double>(op, 1, tensor_accessor);
    const auto step_val = get_input_const_data_as<TRShape, double>(op, 2, tensor_accessor);

    double start = 0;
    double stop = 0;
    double step = 0;

    if (start_val) {
        NODE_VALIDATION_CHECK(op, start_val->size() == 1);
        start = (*start_val)[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(start) && !std::isnan(start), "'start' cannot be nan or infinite.");
    }

    if (stop_val) {
        NODE_VALIDATION_CHECK(op, stop_val->size() == 1);
        stop = (*stop_val)[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(stop) && !std::isnan(stop), "'stop' cannot be nan or infinite.");
    }

    if (step_val) {
        NODE_VALIDATION_CHECK(op, step_val->size() == 1);
        step = (*step_val)[0];
        if (step_allows_zero)
            NODE_VALIDATION_CHECK(op, std::isfinite(step) && !std::isnan(step), "'step' cannot be nan or infinite.");
        else
            NODE_VALIDATION_CHECK(op,
                                  std::isfinite(step) && !std::isnan(step) && step != 0,
                                  "'step' cannot be zero, nan, or infinite.");
    }

    auto output_shapes = std::vector<TRShape>(1);
    if (start_val && stop_val && step_val) {
        // all inputs must be casted to output_type before
        // the rounding for casting values are done towards zero
        if (output_is_integral) {
            start = std::trunc(start);
            stop = std::trunc(stop);
            step = std::trunc(step);
        }

        // the number of elements is: max(ceil((stop − start) / step), 0)
        double span;
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            span = 0;
        } else {
            span = stop - start;
        }

        double strided = ceil(fabs(span) / fabs(step));

        output_shapes[0] = TRShape{static_cast<uint32_t>(strided)};
    } else {
        output_shapes[0] = ov::PartialShape::dynamic(1);
    }
    return output_shapes;
}
}  // namespace ShapeInferRange

namespace v0 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Range* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return ShapeInferRange::range_shape_infer(op,
                                              input_shapes,
                                              op->get_input_element_type(0).is_integral_number(),
                                              false,
                                              tensor_accessor);
}
}  // namespace v0

namespace v4 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Range* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return ShapeInferRange::range_shape_infer(op,
                                              input_shapes,
                                              op->get_output_type().is_integral_number(),
                                              true,
                                              tensor_accessor);
}
}  // namespace v4
}  // namespace op
}  // namespace ov
