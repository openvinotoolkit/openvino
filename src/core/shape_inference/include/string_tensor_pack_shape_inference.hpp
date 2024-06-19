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
    auto output_shapes = std::vector<TRShape>(1);
    const auto& begin_inidices_shape = input_shapes[0];
    if (begin_inidices_shape.is_static()) {
        output_shapes[0] = begin_inidices_shape;
    }
    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
