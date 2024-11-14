// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/fake_convert.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v13 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const FakeConvert* op, const std::vector<T>& input_shapes) {
    const auto inputs_count = input_shapes.size();
    const auto has_shifts_input = inputs_count == 3;
    NODE_VALIDATION_CHECK(op, inputs_count == 2 || has_shifts_input);
    const auto& scales_shape = input_shapes[1];
    if (has_shifts_input) {
        const auto& shifts_shape = input_shapes[2];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               scales_shape.compatible(shifts_shape),
                               "FakeConvert scale shape is not compatible with shift shape.");
    }
    auto output_shapes = std::vector<TRShape>{input_shapes[0]};
    auto& output_pshape = output_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           T::broadcast_merge_into(output_pshape, scales_shape, op::AutoBroadcastType::NUMPY),
                           "Argument shapes are inconsistent.");
    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        input_shapes[0].compatible(output_pshape),
        "FakeConvert support only unidirectional broadcasting, inputs cannot be broadcast into data.");
    return output_shapes;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
