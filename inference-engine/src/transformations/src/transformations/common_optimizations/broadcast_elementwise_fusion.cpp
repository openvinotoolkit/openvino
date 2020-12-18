// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BroadcastElementwiseFusion, "BroadcastElementwiseFusion", 0);

bool is_eliminate_broadcast(const ngraph::PartialShape & input_shape, const ngraph::PartialShape & broadcast_shape) {
    if (input_shape.rank().is_dynamic() || broadcast_shape.rank().is_dynamic()) {
        return false;
    }

    const int64_t & input_shape_rank = input_shape.rank().get_length();
    const int64_t & broadcast_shape_rank = broadcast_shape.rank().get_length();
    if (broadcast_shape_rank > input_shape_rank) {
        //We can not eliminate broadcast op because
        //in the case input_shape will be broadcasted
        return false;
    }
    for (int64_t i_dim = input_shape_rank - 1, b_dim = broadcast_shape_rank - 1; i_dim >= 0 && b_dim >=0; --i_dim, --b_dim) {
        if (input_shape[i_dim].is_static() && broadcast_shape[b_dim].is_static()) {
            const auto &input_shape_dim = input_shape[i_dim].get_length();
            const auto &broadcast_shape_dim = broadcast_shape[b_dim].get_length();
            if (input_shape_dim != broadcast_shape_dim && broadcast_shape_dim != 1) {
                //We can not eliminate broadcast op because
                //input_shape will be broadcast
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

ngraph::pass::BroadcastElementwiseFusion::BroadcastElementwiseFusion() {
    auto broadcast_input = pattern::any_input();
    auto broadcast = pattern::wrap_type<ngraph::opset5::Broadcast>({broadcast_input, pattern::any_input()});
    auto eltwise_input = pattern::any_input();
    auto eltwise = pattern::wrap_type<op::util::BinaryElementwiseArithmetic>({eltwise_input, broadcast});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_value = m.get_pattern_value_map();

        const auto & m_eltwise_input = pattern_value.at(eltwise_input);
        const auto & m_eltwise = pattern_value.at(eltwise_input);

        const auto & m_broadcast_input = pattern_value.at(broadcast_input);
        auto & m_broadcast = pattern_value.at(broadcast);

        if (!is_eliminate_broadcast(m_eltwise_input.get_partial_shape(),
                                    m_broadcast.get_partial_shape())) {
            return false;
        }

        copy_runtime_info(m_broadcast.get_node_shared_ptr(), m_eltwise.get_node_shared_ptr());
        m_broadcast.replace(m_broadcast_input);

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "BroadcastElementwiseFusion");
    register_matcher(m, callback);
}
