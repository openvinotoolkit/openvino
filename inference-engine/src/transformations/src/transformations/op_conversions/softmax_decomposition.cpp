// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/op_conversions/softmax_decomposition.hpp>

#include <memory>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SoftmaxDecomposition, "SoftmaxDecomposition", 0);

ngraph::pass::SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax = pattern::wrap_type<ngraph::opset8::Softmax>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<opset8::Softmax>(m.get_match_root());
        if (!node || transformation_callback(node)) {
            return false;
        }

        auto input = node->input_value(0);
        auto axis = opset8::Constant::create(element::i64, Shape{1}, {node->get_axis()});
        auto reduce_max = std::make_shared<opset8::ReduceMax>(input, axis, true);
        auto sub = std::make_shared<opset8::Subtract>(input, reduce_max);
        auto exp = std::make_shared<opset8::Exp>(sub);
        auto reduce_sum = std::make_shared<opset8::ReduceSum>(exp, axis, true);
        auto div = std::make_shared<opset8::Divide>(exp, reduce_sum);

        replace_node(node, div);
        copy_runtime_info(node, {reduce_max, reduce_sum, sub, exp, div});
        div->set_friendly_name(node->get_friendly_name());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(softmax, matcher_name);
    register_matcher(m, callback);
}
