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

    auto& result_shape = output_shapes[0];
    if (op->get_auto_broadcast().m_type == op::AutoBroadcastType::PDPD) {
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
        then_else_merged = T::broadcast_merge_into(result_shape, input_shapes[src_idx], op->get_auto_broadcast());

        // Try the other way if needed
        if (!then_else_merged && try_bidirection) {
            result_shape = input_shapes[src_idx];
            then_else_merged = T::broadcast_merge_into(result_shape, input_shapes[dest_idx], op->get_auto_broadcast());
        }

        NODE_VALIDATION_CHECK(op,
                              then_else_merged,
                              "'Else' tensor and `Then` tensor are not broadcasted to each other ");
        NODE_VALIDATION_CHECK(op,
                              T::broadcast_merge_into(result_shape, input_shapes[0], op->get_auto_broadcast()),
                              "'Cond' tensor shape is not broadcastable.");
    } else {
        result_shape = input_shapes[2];
        for (int i = 1; i >= 0; i--) {
            if (op->get_auto_broadcast().m_type == op::AutoBroadcastType::NONE) {
                NODE_VALIDATION_CHECK(op,
                                      T::merge_into(result_shape, input_shapes[i]),
                                      "Argument shapes are inconsistent.");
            } else if (op->get_auto_broadcast().m_type == op::AutoBroadcastType::NUMPY) {
                NODE_VALIDATION_CHECK(op,
                                      T::broadcast_merge_into(result_shape, input_shapes[i], op->get_auto_broadcast()),
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