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
    auto input_m = any_input();
    auto logSoftmax_m = wrap_type<ov::opset8::LogSoftmax>({input_m});
    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto logSoftmaxNode = ov::as_type_ptr<ov::opset8::LogSoftmax>(pattern_map.at(logSoftmax_m).get_node_shared_ptr());
        auto axis = std::make_shared<opset8::Constant>(ov::element::i64, ov::Shape{}, logSoftmaxNode->get_axis());
        auto input = pattern_map.at(input_m);
        auto xMax = std::make_shared<opset8::ReduceMax>(input, axis, true);
        auto subtract = std::make_shared<opset8::Subtract>(input, xMax);
        auto exp = std::make_shared<opset8::Exp>(subtract);
        auto s = std::make_shared<opset8::ReduceSum>(exp, axis, true);
        auto log = std::make_shared<opset8::Log>(s);
        auto result = std::make_shared<opset8::Subtract>(subtract, log);
        result->set_friendly_name(logSoftmaxNode->get_friendly_name());
        ov::copy_runtime_info(logSoftmaxNode, {xMax, subtract, exp, s, log, result});
        ov::replace_node(logSoftmaxNode, result);
        return true;
    };

    auto m = std::make_shared<Matcher>(logSoftmax_m, matcher_name);
    this->register_matcher(m, callback);
}
