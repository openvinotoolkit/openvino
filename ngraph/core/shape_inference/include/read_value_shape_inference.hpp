// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/read_value.hpp>

namespace ov {
namespace op {
namespace v3 {
template <class T>
void shape_infer(ReadValue* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    output_shapes[0] = input_shape;
}
}  // namespace v3
}  // namespace op
}  // namespace ov