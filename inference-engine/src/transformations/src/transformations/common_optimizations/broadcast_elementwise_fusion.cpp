// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BroadcastElementwiseFusion, "BroadcastElementwiseFusion", 0);
bool сompatible_shapeы(const ngraph::PartialShape & input_shape1, const ngraph::PartialShape & input_shape1) {

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
        auto broadcast_output_shape = broadcast_node->get_output_partial_shape(0);//output(0).get_partial_shape();
        if (elementwise_input_shape != broadcast_output_shape) {
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
