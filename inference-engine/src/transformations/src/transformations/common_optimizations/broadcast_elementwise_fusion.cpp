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
    if(broadcast_shape_rank > input_shape_rank) {
        //We can not eliminate broadcast op because
        //in the case input_shape will be broadcast
        return false;
    }
    for (int64_t i_dim = 0; i_dim < input_shape_rank; i_dim++) {
        if(input_shape[i_dim].is_static() && broadcast_shape[i_dim].is_static()) {
            const auto & input_shape_dim = input_shape[i_dim].get_length();
            const auto & broadcast_shape_dim = broadcast_shape[i_dim].get_length();
            if (input_shape_dim != broadcast_shape_dim && broadcast_shape_dim != 1) {
                //We can not eliminate broadcast op because
                //input_shape will be broadcast
                return false;
            }
        }
        else if(input_shape[i_dim].is_dynamic() &&
                broadcast_shape[i_dim].is_static() &&
                broadcast_shape[i_dim].get_length() != 1) {
            return false;
        }
    }
    return true;
}

ngraph::pass::BroadcastElementwiseFusion::BroadcastElementwiseFusion() {
    auto input1 = ngraph::pattern::any_input();
    auto input2 = ngraph::pattern::any_input();
    auto broadcast = ngraph::pattern::wrap_type<ngraph::opset5::Broadcast>({input2, pattern::any_input()});
    auto elementwise = ngraph::pattern::wrap_type<ngraph::op::util::BinaryElementwiseArithmetic>({input1, broadcast});

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto &pattern_value = m.get_pattern_value_map();
        auto broadcast_node = pattern_value[broadcast].get_node_shared_ptr();
        auto elementwise_input_node = pattern_value[input1].get_node_shared_ptr();
        if (!broadcast_node) {
            return false;
        }

        auto elementwise_input_shape = elementwise_input_node->get_output_partial_shape(0);
        auto broadcast_output_shape = broadcast_node->get_output_partial_shape(0);
        if (!is_eliminate_broadcast(elementwise_input_shape, broadcast_output_shape)) {
            return false;
        }

        auto input = broadcast_node->input_value(0);
        copy_runtime_info(broadcast_node, input.get_node_shared_ptr());
        broadcast_node->output(0).replace(input);

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(elementwise, "BroadcastElementwiseFusion");
    register_matcher(m, matcher_pass_callback);
}
