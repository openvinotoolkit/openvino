// Copyright (C) 2018-2023 Intel Corporation
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
    if (has_shifts_input) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[1].compatible(input_shapes[2]),
                               "FakeConvert scale shape is not compatible with shift shape.");
    }
    TRShape output_pshape = input_shapes[0];
    NODE_VALIDATION_CHECK(op,
                          TRShape::broadcast_merge_into(output_pshape, input_shapes[1], op::AutoBroadcastType::NUMPY),
                          "Argument shapes are inconsistent.");
    OPENVINO_ASSERT(input_shapes[0].compatible(output_pshape),
                    "FakeConvert support only unidirectional broadcasting, inputs cannot be broadcastd into data.");
    return {output_pshape};
}
}  // namespace v13
}  // namespace op
}  // namespace ov
