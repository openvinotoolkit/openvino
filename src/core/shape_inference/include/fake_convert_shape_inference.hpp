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
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 || input_shapes.size() == 3);
    if (input_shapes.size() == 3) {
        OPENVINO_ASSERT(input_shapes[1].compatible(input_shapes[2]),
                        "FakeConvert scale shape: ",
                        input_shapes[1],
                        " is not compatible with shift shape: ",
                        input_shapes[2]);
    }
    TRShape data_pshape = input_shapes[0];
    NODE_VALIDATION_CHECK(
        op,
        PartialShape::broadcast_merge_into(data_pshape, input_shapes[1], op::AutoBroadcastType::NUMPY),
        "Argument shapes are inconsistent.");
    OPENVINO_ASSERT(input_shapes[0].compatible(data_pshape),
                    "FakeConvert support only unidirectional broadcasting, inputs cannot be broadcastd into data.");
    return {data_pshape};
}
}  // namespace v13
}  // namespace op
}  // namespace ov
