// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/range.hpp>

#include "utils.hpp"
namespace ov {
namespace op {

namespace ShapeInferRange {

template <class T>
inline bool get_data_as_double(
    size_t idx,
    const ov::Node* op,
    std::vector<double>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<double>();
    } else {
        const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
        NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
        axes_value = constant->cast_vector<double>();
    }
    return true;
}

template <>
inline bool get_data_as_double<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    std::vector<double>& axes_value,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        axes_value = ov::opset1::Constant(constant_data.at(idx)).cast_vector<double>();
    } else if (const auto& constant = ov::get_constant_from_source(op->input_value(idx))) {
        axes_value = constant->cast_vector<double>();
    } else {
        return false;
    }
    return true;
}

template <class T>
void range_shape_infer(const Node* op,
                       const std::vector<T>& input_shapes,
                       std::vector<T>& output_shapes,
                       bool output_is_integral,
                       bool step_allows_zero,
                       const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3) && output_shapes.size() == 1);

    NODE_VALIDATION_CHECK(op, input_shapes[0].rank().compatible(0), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[1].rank().compatible(0), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[2].rank().compatible(0), "'step' input is not a scalar");

    std::vector<double> start_val;
    std::vector<double> stop_val;
    std::vector<double> step_val;

    double start = 0;
    double stop = 0;
    double step = 0;

    if (get_data_as_double<T>(0, op, start_val, constant_data)) {
        NODE_VALIDATION_CHECK(op, start_val.size() == 1);
        start = start_val[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(start) && !std::isnan(start), "'start' cannot be nan or infinite.");
    }

    if (get_data_as_double<T>(1, op, stop_val, constant_data)) {
        NODE_VALIDATION_CHECK(op, stop_val.size() == 1);
        stop = stop_val[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(stop) && !std::isnan(stop), "'stop' cannot be nan or infinite.");
    }

    if (get_data_as_double<T>(2, op, step_val, constant_data)) {
        NODE_VALIDATION_CHECK(op, step_val.size() == 1);
        step = step_val[0];
        if (step_allows_zero)
            NODE_VALIDATION_CHECK(op, std::isfinite(step) && !std::isnan(step), "'step' cannot be nan or infinite.");
        else
            NODE_VALIDATION_CHECK(op,
                                  std::isfinite(step) && !std::isnan(step) && step != 0,
                                  "'step' cannot be zero, nan, or infinite.");
    }

    if (start_val.size() == 1 && stop_val.size() == 1 && step_val.size() == 1) {
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

        output_shapes[0] = T{static_cast<uint32_t>(strided)};
    } else {
        output_shapes[0] = ov::PartialShape::dynamic(1);
    }
}
}  // namespace ShapeInferRange

namespace v0 {

template <class T>
void shape_infer(const Range* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    ShapeInferRange::range_shape_infer(op,
                                       input_shapes,
                                       output_shapes,
                                       op->get_input_element_type(0).is_integral_number(),
                                       false,
                                       constant_data);
}

}  // namespace v0

namespace v4 {

template <class T>
void shape_infer(const Range* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    ShapeInferRange::range_shape_infer(op,
                                       input_shapes,
                                       output_shapes,
                                       op->get_output_type().is_integral_number(),
                                       true,
                                       constant_data);
}

}  // namespace v4
}  // namespace op
}  // namespace ov
