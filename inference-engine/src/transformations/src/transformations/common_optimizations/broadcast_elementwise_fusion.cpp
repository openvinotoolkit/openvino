// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BroadcastElementwiseFusion, "BroadcastElementwiseFusion", 0);

bool is_broadcastable_shapes(const ngraph::PartialShape & input_shape1, const ngraph::PartialShape & input_shape2) {
    if (input_shape1.rank().is_dynamic() || input_shape2.rank().is_dynamic()) {
        return false;
    }

    const int64_t & input1_shape_rank = input_shape1.rank().get_length();
    const int64_t & input2_shape_rank = input_shape2.rank().get_length();
    int64_t count_shape_rank = input1_shape_rank;
    if(input1_shape_rank < input2_shape_rank) {
        count_shape_rank = input2_shape_rank;
    }

    for (int64_t i_dim = 1; i_dim <= count_shape_rank; i_dim++) {
        if (input_shape1[input1_shape_rank - i_dim].is_static() && input_shape2[input2_shape_rank - i_dim].is_static()) {
            if (i_dim > input1_shape_rank || i_dim > input2_shape_rank) {
                break;
            }

            if (input_shape1[input1_shape_rank - i_dim].is_static() &&
                input_shape2[input2_shape_rank - i_dim].is_static()) {
                const auto & input1_dim = input_shape1[input1_shape_rank - i_dim].get_length();
                const auto & input2_dim = input_shape2[input2_shape_rank - i_dim].get_length();
                if (input1_dim != input2_dim && input1_dim != 1 && input2_dim != 1) {
                    // this dimensions are not broadcastable
                    return false;
                }
            }
        }
        else if(input_shape1[i_dim].is_dynamic() && input_shape2[i_dim] == 1 ||
                input_shape2[i_dim].is_dynamic() && input_shape1[i_dim] == 1) {
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
        if (!is_broadcastable_shapes(elementwise_input_shape, broadcast_output_shape)) {
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
