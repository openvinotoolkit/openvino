// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_utils.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "util.hpp"  // isPastKeyValuesKey, isPastKeyValuesValue

namespace ov {
namespace npuw {
namespace function {

// ----------------------------------------------------------------------------
// find_mask_parameter
// ----------------------------------------------------------------------------

std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node) {
    if (!add_node || add_node->get_input_size() < 2) {
        return nullptr;
    }

    // Traverse the Add node's mask input (input 1) upwards to find the proper Parameter.
    // Only unary ops are allowed along the way.
    auto mask_in_node = add_node->input(1).get_source_output().get_node_shared_ptr();
    while (mask_in_node && !ov::op::util::is_parameter(mask_in_node)) {
        if (mask_in_node->inputs().size() != 1) {
            LOG_WARN("Non-unary or disconnected op on the way from Add to input mask");
            return nullptr;
        }
        mask_in_node = mask_in_node->inputs()[0].get_source_output().get_node_shared_ptr();
    }

    if (mask_in_node && ov::op::util::is_parameter(mask_in_node)) {
        return std::static_pointer_cast<ov::op::v0::Parameter>(mask_in_node);
    }

    return nullptr;
}

// ----------------------------------------------------------------------------
// find_sdpa_pattern_nodes
// ----------------------------------------------------------------------------

SDPAPatternNodes find_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model) {
    SDPAPatternNodes pattern_nodes;

    // Collect all past key / value parameter nodes (supports multiple blocks
    // after SplitKVCacheIntoBlocks transformation).
    for (auto input : model->inputs()) {
        auto input_node = input.get_node();
        auto input_name = input_node->get_friendly_name();
        if (ov::npuw::util::isPastKeyValuesKey(input_name)) {
            pattern_nodes.past_key_param_nodes.push_back(input_node->shared_from_this());
        } else if (ov::npuw::util::isPastKeyValuesValue(input_name)) {
            pattern_nodes.past_value_param_nodes.push_back(input_node->shared_from_this());
        }
    }

    // Helper: walk backwards from a MatMul input through Reshape/Broadcast/Unsqueeze
    // intermediates until a Concat node is found (or the chain ends).
    auto find_concat_from_matmul = [](const std::shared_ptr<ov::Node>& matmul_node,
                                      size_t input_idx) -> std::shared_ptr<ov::Node> {
        if (!matmul_node)
            return nullptr;

        auto current_node = matmul_node->input(input_idx).get_source_output().get_node_shared_ptr();

        while (current_node) {
            if (ov::is_type<ov::op::v0::Concat>(current_node)) {
                return current_node;
            }

            if (ov::is_type<ov::op::v1::Reshape>(current_node) || ov::is_type<ov::op::v3::Broadcast>(current_node) ||
                ov::is_type<ov::op::v0::Unsqueeze>(current_node)) {
                if (current_node->get_input_size() > 0) {
                    current_node = current_node->input(0).get_source_output().get_node_shared_ptr();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        return nullptr;
    };

    // Search for the canonical pattern: MatMul(Q,K) -> Add(mask) -> Softmax -> MatMul(·,V)
    auto ops = model->get_ordered_ops();
    for (auto&& node : ops) {
        if (!ov::is_type<ov::op::v8::Softmax>(node)) {
            continue;
        }

        pattern_nodes.softmax_node = node;

        // Softmax <- Add?
        auto softmax_input = node->input(0).get_source_output().get_node_shared_ptr();
        if (ov::is_type<ov::op::v1::Add>(softmax_input)) {
            pattern_nodes.add_node = softmax_input;

            // Add <- MatMul (first)?
            auto add_input0 = pattern_nodes.add_node->input(0).get_source_output().get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::MatMul>(add_input0)) {
                pattern_nodes.matmul1_node = add_input0;
                // Find past_key Concat (input 1 of MatMul1)
                pattern_nodes.past_key_concat_node = find_concat_from_matmul(pattern_nodes.matmul1_node, 1);
            }
        }

        // Softmax -> MatMul (second)?
        for (auto&& output : node->outputs()) {
            for (auto&& target_input : output.get_target_inputs()) {
                auto target_node = target_input.get_node()->shared_from_this();
                if (ov::is_type<ov::op::v0::MatMul>(target_node)) {
                    pattern_nodes.matmul2_node = target_node;
                    // Find past_value Concat (input 1 of MatMul2)
                    pattern_nodes.past_value_concat_node = find_concat_from_matmul(pattern_nodes.matmul2_node, 1);
                    break;
                }
            }
            if (pattern_nodes.matmul2_node)
                break;
        }

        if (pattern_nodes.is_valid()) {
            pattern_nodes.log_pattern();
            break;
        }
    }

    return pattern_nodes;
}

}  // namespace function
}  // namespace npuw
}  // namespace ov
