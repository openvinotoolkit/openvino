// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/graph_util.hpp>
#include <openvino/op/assign.hpp>

#include "copy_shape_inference.hpp"
#include "utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Assign* op, const std::vector<TShape>& input_shapes) {
    auto output_shapes = ov::op::copy_shape_infer(op, input_shapes);

    const auto& input_shape = input_shapes[0];
    const auto& variable_info = op->get_variable()->get_info();
    NODE_VALIDATION_CHECK(op,
                          op->get_variable_id() == variable_info.variable_id,
                          "Variables identifiers are inconsistent.");
    const auto& arg_t = op->get_input_element_type(0);
    NODE_VALIDATION_CHECK(op, arg_t == variable_info.data_type, "Variables types are inconsistent.");

    if (input_shape.is_static() && variable_info.data_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_shape.to_shape() == variable_info.data_shape.to_shape(),
                              "Variables output shapes are inconsistent.");
    }
    return output_shapes;
}
}  // namespace v3
}  // namespace op
}  // namespace ov
