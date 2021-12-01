// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/range.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v4 {

template <class T>
inline std::shared_ptr<ov::opset1::Constant> get_data_as_const(
    size_t idx,
    const ov::Node* op,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    if (constant_data.count(idx)) {
        return std::make_shared<ov::opset1::Constant>(constant_data.at(idx));
    }
    const auto& constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(idx));
    NODE_VALIDATION_CHECK(op, constant != nullptr, "Static shape inference lacks constant data on port ", idx);
    return constant;
}

template <>
inline std::shared_ptr<ov::opset1::Constant> get_data_as_const<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        return std::make_shared<ov::opset1::Constant>(constant_data.at(idx));
    }

    return ov::get_constant_from_source(op->input_value(idx));
}

template <class T>
void shape_infer(const Range* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3) && output_shapes.size() == 1);

    NODE_VALIDATION_CHECK(op, input_shapes[0].compatible(T{}), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[1].compatible(T{}), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(op, input_shapes[2].compatible(T{}), "'step' input is not a scalar");

    auto const_start = get_data_as_const<T>(0, op, constant_data);
    auto const_stop = get_data_as_const<T>(1, op, constant_data);
    auto const_step = get_data_as_const<T>(2, op, constant_data);

    double start = 0;
    double stop = 0;
    double step = 0;

    if (const_start != nullptr) {
        std::vector<double> start_val = const_start->template cast_vector<double>();
        NODE_VALIDATION_CHECK(op, start_val.size() == 1);
        start = start_val[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(start) && !std::isnan(start), "'start' cannot be nan or infinite.");
    }

    if (const_stop != nullptr) {
        std::vector<double> stop_val = const_stop->template cast_vector<double>();
        NODE_VALIDATION_CHECK(op, stop_val.size() == 1);
        stop = stop_val[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(stop) && !std::isnan(stop), "'stop' cannot be nan or infinite.");
    }

    if (const_step != nullptr) {
        std::vector<double> step_val = const_step->template cast_vector<double>();
        NODE_VALIDATION_CHECK(op, step_val.size() == 1);
        step = step_val[0];
        NODE_VALIDATION_CHECK(op, std::isfinite(step) && !std::isnan(step), "'step' cannot be nan or infinite.");
    }

    auto output_type = op->get_output_type();
    if (const_start != nullptr && const_stop != nullptr && const_step != nullptr) {
        // all inputs must be casted to output_type before
        // the rounding for casting values are done towards zero
        if (output_type.is_integral_number() && const_start->get_output_element_type(0).is_real()) {
            start = std::trunc(start);
        }
        if (output_type.is_integral_number() && const_stop->get_output_element_type(0).is_real()) {
            stop = std::trunc(stop);
        }
        if (output_type.is_integral_number() && const_step->get_output_element_type(0).is_real()) {
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
}  // namespace v4
}  // namespace op
}  // namespace ov
