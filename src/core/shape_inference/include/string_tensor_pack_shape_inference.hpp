// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/op/string_tensor_pack.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const StringTensorPack* op,
                                 const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
    const auto begins_shape = input_shapes[0];
    const auto ends_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, begins_shape == ends_shape);
    auto output_shapes = std::vector<TRShape>{begins_shape};
    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
