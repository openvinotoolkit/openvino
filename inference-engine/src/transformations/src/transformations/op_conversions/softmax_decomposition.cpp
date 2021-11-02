// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <transformations/op_conversions/softmax_decomposition.hpp>

#include <memory>
#include <vector>

#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

namespace {
bool decompose_softmax(const std::shared_ptr<ngraph::Node> &node,
                       const int64_t softmax_axis) {
    if (!node)
        return false;

    auto input = node->input_value(0);
    auto axis = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {softmax_axis});
    auto reduce_max = std::make_shared<ngraph::opset8::ReduceMax>(input, axis, true);
    auto sub = std::make_shared<ngraph::opset8::Subtract>(input, reduce_max);
    auto exp = std::make_shared<ngraph::opset8::Exp>(sub);
    auto reduce_sum = std::make_shared<ngraph::opset8::ReduceSum>(exp, axis, true);
    auto div = std::make_shared<ngraph::opset8::Divide>(exp, reduce_sum);

    replace_node(node, div);
    copy_runtime_info(node, {reduce_max, reduce_sum, sub, exp, div});
    div->set_friendly_name(node->get_friendly_name());
    return true;
}
}  // namespace

NGRAPH_RTTI_DEFINITION(ngraph::pass::SoftmaxDecomposition, "SoftmaxDecomposition", 0);

ngraph::pass::SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax = std::make_shared<pattern::op::Or>(ngraph::OutputVector{pattern::wrap_type<ngraph::opset8::Softmax>(),
            pattern::wrap_type<ngraph::opset1::Softmax>()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root()))
            return false;

        if (ov::is_type<ngraph::opset1::Softmax>(pattern_map.at(softmax).get_node_shared_ptr())) {
            auto node = std::dynamic_pointer_cast<ngraph::opset1::Softmax>(softmax);
            return decompose_softmax(node, static_cast<int64_t>(node->get_axis()));
        } else if (ov::is_type<ngraph::opset8::Softmax>(pattern_map.at(softmax).get_node_shared_ptr())) {
            auto node = std::dynamic_pointer_cast<ngraph::opset8::Softmax>(softmax);
            return decompose_softmax(node, node->get_axis());
        } else {
            return false;
        }
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(softmax, matcher_name);
    register_matcher(m, callback);
}
