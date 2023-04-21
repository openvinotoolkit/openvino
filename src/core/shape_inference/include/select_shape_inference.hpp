// Copyright (C) 2018-2023 Intel Corporation
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

    result_shape = input_shapes[2];
    for (int input_port = 1; input_port >= 0; input_port--) {
        if (broadcast_spec.m_type == op::AutoBroadcastType::NONE) {
            NODE_VALIDATION_CHECK(op,
                                  T::merge_into(result_shape, input_shapes[input_port]),
                                  "Argument shapes are inconsistent.");
        } else if (broadcast_spec.m_type == op::AutoBroadcastType::NUMPY ||
                   broadcast_spec.m_type == op::AutoBroadcastType::PDPD) {
            NODE_VALIDATION_CHECK(op,
                                  T::broadcast_merge_into(result_shape, input_shapes[input_port], broadcast_spec),
                                  "Argument shapes are inconsistent.");
        } else {
            NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
        }
    }
}

}  // namespace v1
}  // namespace op
}  // namespace ov
