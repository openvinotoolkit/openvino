// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/inverse.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v14 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Inverse* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto input_rank = input_shape.rank();
    if (input_rank.is_static()) {
        const auto size = input_shape.size();
        NODE_VALIDATION_CHECK(op, size >= 2, "Input must be at least a 2D matrix.");

        if (input_shape[size - 2].is_static() && input_shape[size - 1].is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   input_shape[size - 2].compatible(input_shape[size - 1]),
                                   "Input must contain square matrices of the same shape.");
        }
    }

    return {input_shape};
}
}  // namespace v14
}  // namespace op
}  // namespace ov
