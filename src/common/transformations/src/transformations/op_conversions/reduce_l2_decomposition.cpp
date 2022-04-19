// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/reduce_l2_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ReduceL2Decomposition::ReduceL2Decomposition() {
    MATCHER_SCOPE(ReduceL2Decomposition);
    // decomposes ReduceL2 operations into sqrt(ReduceSum(x * x))
    auto reduce_l2 = ngraph::pattern::wrap_type<opset4::ReduceL2>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reduce_l2_node =
            std::dynamic_pointer_cast<ngraph::opset4::ReduceL2>(pattern_to_output.at(reduce_l2).get_node_shared_ptr());

        if (reduce_l2_node == nullptr || transformation_callback(reduce_l2_node)) {
            return false;
        }

        auto const_2 =
            ngraph::opset4::Constant::create(reduce_l2_node->input_value(0).get_element_type(), Shape{}, {2.0f});
        auto square = std::make_shared<ngraph::opset4::Power>(reduce_l2_node->input_value(0), const_2);
        auto reduce_sum = register_new_node<ngraph::opset4::ReduceSum>(square,
                                                                       reduce_l2_node->input_value(1),
                                                                       reduce_l2_node->get_keep_dims());
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);
        sqrt->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(reduce_l2_node, {sqrt, reduce_sum, square, const_2});
        ngraph::replace_node(m.get_match_root(), sqrt);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce_l2, matcher_name);
    register_matcher(m, callback);
}
