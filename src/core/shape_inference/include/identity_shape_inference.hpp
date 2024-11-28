// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/identity.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Identity* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];

    return {input_shape};
}
}  // namespace v16
}  // namespace op
}  // namespace ov
