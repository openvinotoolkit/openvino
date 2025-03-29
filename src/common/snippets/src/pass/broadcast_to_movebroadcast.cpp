// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/broadcast_to_movebroadcast.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/opsets/opset1.hpp"
#include "openvino/core/rt_info.hpp"


ov::snippets::pass::BroadcastToMoveBroadcast::BroadcastToMoveBroadcast() {
    MATCHER_SCOPE(BroadcastToMoveBroadcast);

    auto m_broadcast = ov::pass::pattern::wrap_type<ov::op::v1::Broadcast, ov::op::v3::Broadcast>();

    auto callback = [](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::BroadcastToMoveBroadcast")
        auto root = m.get_match_root();
        if (auto broadcast_v1 = ov::as_type_ptr<const ov::op::v1::Broadcast>(root)) {
            if (broadcast_v1->get_broadcast_spec().m_type != ov::op::AutoBroadcastType::NUMPY)
                return false;
        } else if (auto broadcast_v3 = ov::as_type_ptr<const ov::op::v3::Broadcast>(root)) {
            if (broadcast_v3->get_broadcast_spec().m_type != ov::op::BroadcastType::NUMPY)
                return false;
        }

        const auto target_shape = root->get_output_partial_shape(0);
        const auto value_shape = root->get_input_partial_shape(0);
        OPENVINO_ASSERT(target_shape.is_static() && value_shape.rank().is_static(), "Broadcast with dynamic target shape is not supported in Snippets");
        // Insert BroadcastMove only if the last dimension needs to be broadcasted. Higher-level dims broadcasting
        // will be handled by pointer arithmetics. Note that this behavior should be changed in case of full op::Boradcast support.
        Output<ov::Node> in_value = root->input_value(0);
        if (*target_shape.rbegin() != *value_shape.rbegin()) {
            auto broadcasted_dim = ov::Dimension(*target_shape.rbegin());
            const auto& broadcast_node = std::make_shared<ov::snippets::op::BroadcastMove>(in_value, broadcasted_dim);
            in_value = broadcast_node->output(0);
        }

        replace_output_update_name(root->output(0), in_value);
        ov::copy_runtime_info(root, in_value.get_node_shared_ptr());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_broadcast, matcher_name);
    register_matcher(m, callback);
}
