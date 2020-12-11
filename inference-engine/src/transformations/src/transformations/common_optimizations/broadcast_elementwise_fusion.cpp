// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

#include <iostream>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BroadcastElementwiseFusion, "BroadcastElementwiseFusion", 0);

ngraph::pass::BroadcastElementwiseFusion::BroadcastElementwiseFusion() {
    auto input1 = ngraph::pattern::any_input();
    auto input2 = ngraph::pattern::any_input();
    auto broadcast = ngraph::pattern::wrap_type<ngraph::opset5::Broadcast>({input2, pattern::any_input()});
    auto elementwise = ngraph::pattern::wrap_type<ngraph::op::util::BinaryElementwiseArithmetic>({input1, broadcast});

    ngraph::graph_rewrite_callback matcher_pass_callback = [](ngraph::pattern::Matcher& m) {
        auto elem = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(m.get_match_root());

        auto elem_input1 = elem->input_value(0);
        auto elem_input2 = elem->input_value(1);

        auto input1_shape = elem->input(0).get_shape();
        auto input2_shape = elem->input(1).get_shape();
        if (input1_shape != input2_shape) {
            return false;
        }

        auto broadcast = elem_input2.get_node_shared_ptr();
        if (!broadcast) {
            return false;
        }
        const auto & channel_dim = elem_input2.get_partial_shape()[1].get_length();
        const auto & rank = elem_input2.get_partial_shape().rank().get_length();

        auto input = broadcast->input_value(0);
        copy_runtime_info(broadcast, input.get_node_shared_ptr());
        broadcast->output(0).replace(broadcast->input_value(0));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(elementwise, "BroadcastElementwiseFusion");
    register_matcher(m, matcher_pass_callback);
}
