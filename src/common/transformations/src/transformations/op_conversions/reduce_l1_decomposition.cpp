// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_l1_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ReduceL1Decomposition::ReduceL1Decomposition() {
    MATCHER_SCOPE(ReduceL1Decomposition);
    // decomposes ReduceL1 operations into ReduceSum(abs(x))
    auto reduce_l1 = ngraph::pattern::wrap_type<opset4::ReduceL1>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reduce_l1_node =
            std::dynamic_pointer_cast<ngraph::opset4::ReduceL1>(pattern_to_output.at(reduce_l1).get_node_shared_ptr());

        if (reduce_l1_node == nullptr || transformation_callback(reduce_l1_node)) {
            return false;
        }

        auto abs = std::make_shared<ngraph::opset4::Abs>(reduce_l1_node->input_value(0));
        auto reduce_sum = register_new_node<ngraph::opset4::ReduceSum>(abs,
                                                                       reduce_l1_node->input_value(1),
                                                                       reduce_l1_node->get_keep_dims());

        reduce_sum->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(reduce_l1_node, {abs, reduce_sum});
        ngraph::replace_node(m.get_match_root(), reduce_sum);
        MATCHER_SCOPE_ENABLE(ReduceL1Decomposition);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_l1, matcher_name);
    register_matcher(m, callback);
}
