// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/assign.hpp>

namespace ov {
namespace op {
namespace v3 {

template <class T1, class T2>
void inline default_work(T1* op, T2& shape) {
    NODE_VALIDATION_CHECK(op, false, "[Assign]Can not infer shape based on input shape");
}

template <class T1>
void inline default_work(T1* op, ov::PartialShape& shape) {
    shape = ov::PartialShape::dynamic();
}

template <class T>
void shape_infer(Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    if (input_shape.is_static())
        output_shapes[0] = input_shapes[0];
    else
        default_work(op, output_shapes[0]);
}
}  // namespace v3

namespace v6 {

template <class T>
void shape_infer(Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    output_shapes[0] = input_shapes[0];
}
}  // namespace v6
}  // namespace op
}  // namespace ov
