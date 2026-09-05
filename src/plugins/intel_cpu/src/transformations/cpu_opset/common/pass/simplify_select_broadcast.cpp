// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simplify_select_broadcast.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu::pass {

namespace {

/**
 * Returns true if the node is a scalar constant (rank 0 or single-element rank-1 with shape {}).
 */
bool is_scalar_constant(const std::shared_ptr<ov::Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
        return false;
    }
    return constant->get_shape().empty();  // rank-0 == scalar
}

/**
 * Attempts to simplify input `broadcast_idx` of the Select node.
 * If the input at `broadcast_idx` is a Broadcast of a scalar constant, replace it with
 * that scalar constant and return true.
 */
bool try_simplify_input(ov::op::v1::Select* select_node,
                        std::size_t broadcast_idx,
                        ov::pass::pattern::Matcher& m) {
    auto& pattern_value_map = m.get_pattern_value_map();
    (void)pattern_value_map;

    const auto broadcast_input = select_node->input_value(broadcast_idx);
    auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(broadcast_input.get_node_shared_ptr());
    if (!broadcast) {
        return false;
    }

    // The Broadcast must be consumed only by this Select.
    if (broadcast->get_output_target_inputs(0).size() != 1) {
        return false;
    }

    // The data input to Broadcast (index 0) must be a scalar constant.
    const auto broadcast_data = broadcast->input_value(0).get_node_shared_ptr();
    if (!is_scalar_constant(broadcast_data)) {
        return false;
    }

    // Replace the Broadcast with the scalar constant.
    copy_runtime_info(broadcast, select_node->shared_from_this());
    select_node->input(broadcast_idx).replace_source_output(broadcast->input_value(0));

    return true;
}

}  // namespace

SimplifySelectBroadcast::SimplifySelectBroadcast() {
    MATCHER_SCOPE(SimplifySelectBroadcast);

    // Match any Select node with NUMPY auto-broadcast.
    auto select_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Select>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        auto select_node_ptr = ov::as_type_ptr<ov::op::v1::Select>(m.get_match_root());
        if (!select_node_ptr) {
            return false;
        }

        // Only handle NUMPY auto-broadcast (default for aten::where exports).
        if (select_node_ptr->get_auto_broadcast() != ov::op::AutoBroadcastType::NUMPY) {
            return false;
        }

        bool changed = false;

        // Check input 1 (on_true) and input 2 (on_false) — both are data inputs.
        // Input 0 is the condition boolean mask; skip it.
        for (std::size_t idx : {std::size_t{1}, std::size_t{2}}) {
            changed |= try_simplify_input(select_node_ptr.get(), idx, m);
        }

        return changed;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(select_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_cpu::pass
