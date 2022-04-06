// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/graph_util.hpp>
#include <openvino/op/assign.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class T>
void shape_infer(const Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    const auto& variable_info = op->m_variable->get_info();
    NODE_VALIDATION_CHECK(op,
                          op->m_variable_id == variable_info.variable_id,
                          "Variables identifiers are inconsistent.");
    const auto& arg_t = op->get_input_element_type(0);
    NODE_VALIDATION_CHECK(op, arg_t == variable_info.data_type, "Variables types are inconsistent.");

    if (input_shape.is_static() && variable_info.data_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              input_shape.to_shape() == variable_info.data_shape.to_shape(),
                              "Variables output shapes are inconsistent.");
    }
    copy_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v3

namespace v6 {

template <class T>
void shape_infer(const Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    copy_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v6
}  // namespace op
}  // namespace ov
