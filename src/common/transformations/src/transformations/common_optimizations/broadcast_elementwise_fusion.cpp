// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

namespace {

bool can_eliminate_broadcast(const ngraph::Output<ngraph::Node>& eltwise,
                             const ngraph::Output<ngraph::Node>& eltwise_input,
                             const ngraph::Output<ngraph::Node>& broadcast) {
    auto b = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(eltwise.get_node_shared_ptr());
    if (!b || b->get_autob() == ngraph::op::AutoBroadcastType::NONE) {
        return false;
    }

    // Check that eltwise_input is the same input which comes to ShapeOf which comes
    // to Broadcast operation as a output shape target. In this case we can eliminate
    // Broadcast since eltwise_input will broadcast another eltwise input automatically.
    auto broadcast_input = broadcast.get_node()->get_input_node_shared_ptr(1);
    if ((ov::is_type<ngraph::opset5::ShapeOf>(broadcast_input) ||
         ov::is_type<ngraph::opset1::ShapeOf>(broadcast_input)) &&
        broadcast_input->input_value(0) == eltwise_input) {
        return true;
    }

    const auto& input_shape = eltwise_input.get_partial_shape();
    const auto& broadcast_shape = broadcast.get_partial_shape();

    if (input_shape.rank().is_dynamic() || broadcast_shape.rank().is_dynamic()) {
        return false;
    }

    const int64_t& input_shape_rank = input_shape.rank().get_length();
    const int64_t& broadcast_shape_rank = broadcast_shape.rank().get_length();
    if (broadcast_shape_rank > input_shape_rank) {
        // We can not eliminate broadcast op because
        // in the case input_shape will be broadcasted
        return false;
    }
    for (int64_t i_dim = input_shape_rank - 1, b_dim = broadcast_shape_rank - 1; i_dim >= 0 && b_dim >= 0;
         --i_dim, --b_dim) {
        if (input_shape[i_dim].is_static() && broadcast_shape[b_dim].is_static()) {
            const auto& input_shape_dim = input_shape[i_dim].get_length();
            const auto& broadcast_shape_dim = broadcast_shape[b_dim].get_length();
            if (input_shape_dim != broadcast_shape_dim && broadcast_shape_dim != 1) {
                // We can not eliminate broadcast op because
                // input_shape will be broadcast
                return false;
            }
        } else if (input_shape[i_dim].is_dynamic() && broadcast_shape[i_dim].is_static() &&
                   broadcast_shape[i_dim].get_length() != 1) {
            return false;
        } else if (broadcast_shape[i_dim].is_dynamic() && input_shape[i_dim].is_static() &&
                   input_shape[i_dim].get_length() == 1) {
            return false;
        } else if (broadcast_shape[i_dim].is_dynamic() && input_shape[i_dim].is_dynamic()) {
            return false;
        }
    }
    return true;
}

}  // namespace

ngraph::pass::BroadcastElementwiseFusion::BroadcastElementwiseFusion() {
    MATCHER_SCOPE(BroadcastElementwiseFusion);
    auto broadcast_input = pattern::any_input();
    auto broadcast = pattern::wrap_type<ngraph::opset5::Broadcast>({broadcast_input, pattern::any_input()},
                                                                   pattern::consumers_count(1));
    auto eltwise_input = pattern::any_input();
    auto eltwise = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({eltwise_input, broadcast});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_value = m.get_pattern_value_map();

        const auto& m_eltwise_input = pattern_value.at(eltwise_input);
        const auto& m_eltwise = pattern_value.at(eltwise);

        const auto& m_broadcast_input = pattern_value.at(broadcast_input);
        auto& m_broadcast = pattern_value.at(broadcast);

        if (!can_eliminate_broadcast(m_eltwise, m_eltwise_input, m_broadcast)) {
            return false;
        }

        copy_runtime_info(m_broadcast.get_node_shared_ptr(), m_eltwise.get_node_shared_ptr());
        m_broadcast.replace(m_broadcast_input);

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, matcher_name);
    register_matcher(m, callback);
}
