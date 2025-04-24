// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/select.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Select* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    const auto& broadcast_spec = op->get_auto_broadcast();
    auto output_shapes = std::vector<TRShape>();
    if (broadcast_spec.m_type == op::AutoBroadcastType::PDPD) {
        output_shapes.push_back(input_shapes[1]);
        auto& result_shape = output_shapes[0];
        // in PDPD type, Broadcast-merging 'else' into 'then' one way not each other.
        NODE_VALIDATION_CHECK(op,
                              TRShape::broadcast_merge_into(result_shape, input_shapes[2], broadcast_spec),
                              "'Else' tensor shape is not broadcastable.");
        NODE_VALIDATION_CHECK(op,
                              TRShape::broadcast_merge_into(result_shape, input_shapes[0], broadcast_spec),
                              "'Cond' tensor shape is not broadcastable.");
    } else {
        output_shapes.push_back(input_shapes[2]);
        auto& result_shape = output_shapes[0];
        for (int input_port = 1; input_port >= 0; input_port--) {
            if (broadcast_spec.m_type == op::AutoBroadcastType::NONE) {
                NODE_VALIDATION_CHECK(op,
                                      TRShape::merge_into(result_shape, input_shapes[input_port]),
                                      "Argument shapes are inconsistent.");
            } else if (broadcast_spec.m_type == op::AutoBroadcastType::NUMPY) {
                NODE_VALIDATION_CHECK(
                    op,
                    TRShape::broadcast_merge_into(result_shape, input_shapes[input_port], broadcast_spec),
                    "Argument shapes are inconsistent.");
            } else {
                NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
            }
        }
    }

    return output_shapes;
}

}  // namespace v1
}  // namespace op
}  // namespace ov
