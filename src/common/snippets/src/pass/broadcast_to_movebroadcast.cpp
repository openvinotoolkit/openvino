// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/broadcast_to_movebroadcast.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <numeric>

using namespace ngraph;

ngraph::snippets::pass::BroadcastToMoveBroadcast::BroadcastToMoveBroadcast() {
    MATCHER_SCOPE(BroadcastToMoveBroadcast);

    auto m_broadcast = ngraph::pattern::wrap_type<ngraph::op::v1::Broadcast, ngraph::op::v3::Broadcast>();

    auto callback = [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::BroadcastToMoveBroadcast")
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

        if (target_shape.is_dynamic() || value_shape.is_dynamic()) {
            return false;
        }

        // Insert MoveBroadcast only in cases only if the last dimension needs to be broadcasted
        // Otherwise we remove Broadcast node
        if (*target_shape.rbegin() != *value_shape.rbegin()) {
            ov::PartialShape broadcasted_shape = value_shape;
            *broadcasted_shape.rbegin() = *target_shape.rbegin();
            const auto move_broadcast = std::make_shared<ngraph::snippets::op::BroadcastMove>(root->input_value(0), broadcasted_shape);
            ov::copy_runtime_info(root, move_broadcast);
            move_broadcast->set_friendly_name(root->get_friendly_name());
            replace_output_update_name(root->output(0), move_broadcast);
        } else {
            replace_output_update_name(root->output(0), root->input_value(0));
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_broadcast, matcher_name);
    register_matcher(m, callback);
}
