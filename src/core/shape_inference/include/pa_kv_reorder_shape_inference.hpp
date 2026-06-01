// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/pa_kv_reorder.hpp"
#include "utils.hpp"

namespace ov::op::internal {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const PaKVReorder* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);

    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(4));
    for (size_t i = 2; i < 6; ++i) {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[i].rank().compatible(1));
    }

    return {TRShape{1}};
}
}  // namespace ov::op::internal
