// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/select.hpp>

namespace ov {
namespace op {
namespace v1 {

template <class T>
void shape_infer(const Select* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);

    const auto& broadcast_spec = op->get_auto_broadcast();
    auto& result_shape = output_shapes[0];
    if (broadcast_spec.m_type == op::AutoBroadcastType::PDPD) {
        result_shape = input_shapes[1];  // 'then' tensor
        // in PDPD type, Broacast-merging 'else' into 'then' one way not each other.
        NODE_VALIDATION_CHECK(op,
                              T::broadcast_merge_into(result_shape, input_shapes[2], broadcast_spec),
                              "'Else' tensor shape is not broadcastable.");
        NODE_VALIDATION_CHECK(op,
                              T::broadcast_merge_into(result_shape, input_shapes[0], broadcast_spec),
                              "'Cond' tensor shape is not broadcastable.");
    } else {
        result_shape = input_shapes[2];
        for (int input_port = 1; input_port >= 0; input_port--) {
            if (broadcast_spec.m_type == op::AutoBroadcastType::NONE) {
                NODE_VALIDATION_CHECK(op,
                                      T::merge_into(result_shape, input_shapes[input_port]),
                                      "Argument shapes are inconsistent.");
            } else if (broadcast_spec.m_type == op::AutoBroadcastType::NUMPY) {
                NODE_VALIDATION_CHECK(op,
                                      T::broadcast_merge_into(result_shape, input_shapes[input_port], broadcast_spec),
                                      "Argument shapes are inconsistent.");
            } else {
                NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
            }
        }
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov