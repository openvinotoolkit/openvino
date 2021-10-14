// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/assign.hpp>
#include "shape_infer_utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class T>
void shape_infer(const Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
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
