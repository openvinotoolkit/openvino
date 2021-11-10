// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/graph_util.hpp>
#include <openvino/op/assign.hpp>

#include "shape_infer_utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class T1, class T2>
bool is_equal(const T1& lhs, const T2& rhs) {
    OPENVINO_ASSERT(lhs.is_static() && rhs.is_static());
    OPENVINO_ASSERT(lhs.rank().get_length() == rhs.rank().get_length());
    const auto rank = lhs.rank().get_length();
    bool ret = true;
    for (size_t i = 0; i < rank; i++) {
        if (lhs[i].get_length() != rhs[i].get_length())
            ret = false;
    }
    return ret;
}

template <class T>
bool is_equal(const T& lhs, const T& rhs) {
    return lhs == rhs;
}

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
                              is_equal(input_shape, variable_info.data_shape),
                              "Variables output shapes are inconsistent.");
    }
    if (input_shape.is_static())
        output_shapes[0] = input_shapes[0];
    else
        ShapeInfer::default_work(output_shapes[0]);
}
}  // namespace v3

namespace v6 {

template <class T>
void shape_infer(const Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    output_shapes[0] = input_shapes[0];
}
}  // namespace v6
}  // namespace op
}  // namespace ov
