// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_logsoftmax.hpp"
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ov::intel_cpu::ConvertLogSoftmax::ConvertLogSoftmax() {
    MATCHER_SCOPE(ConvertLogSoftmax);
    using namespace ov::pass::pattern;
    auto logSoftmax = wrap_type<ov::opset8::LogSoftmax>({any_input()});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto logSoftmaxNode = ov::as_type_ptr<ov::opset8::LogSoftmax>(pattern_map.at(logSoftmax).get_node_shared_ptr());
        auto axis = std::make_shared<opset8::Constant>(ov::element::i64, ov::Shape{}, logSoftmaxNode->get_axis());
        auto input = logSoftmaxNode->get_input_node_ptr(0);
        auto xMax = std::make_shared<opset8::ReduceMax>(input->get_default_output(), axis->get_default_output(), true);
        auto subtract = std::make_shared<opset8::Subtract>(input->get_default_output(), xMax->get_default_output());
        auto tmp = std::make_shared<opset8::Exp>(subtract->get_default_output());
        auto s = std::make_shared<opset8::ReduceSum>(tmp->get_default_output(), axis->get_default_output(), true);
        auto log = std::make_shared<opset8::Log>(s);
        auto result = std::make_shared<opset8::Subtract>(subtract, log);
        result->set_friendly_name(logSoftmaxNode->get_friendly_name());
        ov::replace_node(logSoftmaxNode, result);
        return true;
    };

    auto m = std::make_shared<Matcher>(logSoftmax, matcher_name);
    this->register_matcher(m, callback);
}
