// Copyright (C) 2018-2021 Intel Corporation
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
        // PDPD broadcast rule is one-way broadcast. Needs to figure out src and dest
        // when `then` tensor and `else` tensor are broadcasted to each other.
        size_t dest_idx = 1;
        auto try_bidirection = false;
        auto then_else_merged = false;

        if (input_shapes[1].rank().is_static() && input_shapes[2].rank().is_static()) {
            auto input1_rank = input_shapes[1].size();
            auto input2_rank = input_shapes[2].size();

            // When rank is same, need to try broadcast merging after exchange src and dest if default src can't merge
            // into dest;
            if (input2_rank == input1_rank)
                try_bidirection = true;
            // bigger rank is dest and the other is src;
            else
                dest_idx = input1_rank > input2_rank ? 1 : 2;
        }

        size_t src_idx = (dest_idx == 1) ? 2 : 1;
        result_shape = input_shapes[dest_idx];

        // Try broadcast merging src into dest
        then_else_merged = T::broadcast_merge_into(result_shape, input_shapes[src_idx], broadcast_spec);

        // Try the other way if needed
        if (!then_else_merged && try_bidirection) {
            result_shape = input_shapes[src_idx];
            then_else_merged = T::broadcast_merge_into(result_shape, input_shapes[dest_idx], broadcast_spec);
        }

        NODE_VALIDATION_CHECK(op,
                              then_else_merged,
                              "'Else' tensor and `Then` tensor are not broadcasted to each other ");
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