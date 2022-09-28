// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/read_value.hpp>

#include "utils.hpp"
namespace ov {
namespace op {

template <class OpType, class ShapeType>
void read_value_shape_infer(const OpType* op,
                            const std::vector<ShapeType>& input_shapes,
                            std::vector<ShapeType>& output_shapes) {
    copy_shape_infer(op, input_shapes, output_shapes);
}

namespace v3 {
template <class T>
void shape_infer(const ReadValue* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    read_value_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v3

namespace v6 {
template <class T>
void shape_infer(const ReadValue* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    read_value_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v6
}  // namespace op
}  // namespace ov