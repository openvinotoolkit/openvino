// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/log_softmax_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::LogSoftmaxDecomposition::LogSoftmaxDecomposition() {
    MATCHER_SCOPE(LogSoftmaxDecomposition);
    // Decomposes LogSoftmax(x, axis) op into sub-graph x - log(reduce_sum(exp(x), axis))
    auto log_softmax = ngraph::pattern::wrap_type<opset5::LogSoftmax>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto log_softmax_node = std::dynamic_pointer_cast<ngraph::opset5::LogSoftmax>(
            pattern_to_output.at(log_softmax).get_node_shared_ptr());

        if (log_softmax_node == nullptr || transformation_callback(log_softmax_node)) {
            return false;
        }

        auto axis1 =
            ngraph::opset5::Constant::create(element::Type_t::i64, ngraph::Shape{1}, {log_softmax_node->get_axis()});
        auto axis2 =
            ngraph::opset5::Constant::create(element::Type_t::i64, ngraph::Shape{1}, {log_softmax_node->get_axis()});
        auto max = std::make_shared<ngraph::opset5::ReduceMax>(log_softmax_node->input_value(0), axis1, true);
        auto sub = std::make_shared<ngraph::opset5::Subtract>(log_softmax_node->input_value(0), max);
        auto exp = std::make_shared<ngraph::opset5::Exp>(sub);
        auto sum = std::make_shared<ngraph::opset5::ReduceSum>(exp, axis2, true);
        auto log = std::make_shared<ngraph::opset5::Log>(sum);
        auto sub_end = std::make_shared<ngraph::opset5::Subtract>(sub, log);

        sub_end->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(log_softmax_node, {axis1, axis2, max, sub, exp, sum, log, sub_end});
        ngraph::replace_node(m.get_match_root(), sub_end);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(log_softmax, matcher_name);
    register_matcher(m, callback);
}
