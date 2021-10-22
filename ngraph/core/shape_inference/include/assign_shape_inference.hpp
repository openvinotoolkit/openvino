// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/assign.hpp>

namespace ov {
namespace op {
template <class T1, class T2>
void inline default_work(T1* op, T2& shape) {
    NODE_VALIDATION_CHECK(op, false, "[Assign]Can not infer shape based on input shape");
}

template <class T1>
void inline default_work(T1* op, ov::PartialShape& shape) {
    shape = ov::PartialShape::dynamic();
}

template <class T1, class T2>
void assign_shape_infer(T1* op, const std::vector<T2>& input_shapes, std::vector<T2>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    if (input_shape.is_static())
        output_shapes[0] = input_shapes[0];
    else
        default_work(op, output_shapes[0]);
}

namespace v3 {

template <class T>
void shape_infer(Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    assign_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v3

namespace v6 {

template <class T>
void shape_infer(Assign* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    assign_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v6
}  // namespace op
}  // namespace ov
