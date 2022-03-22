// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_subtract.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/core/validation_util.hpp>

#include "itt.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

static bool convert_subtract(const std::shared_ptr<ngraph::Node>& node) {
    auto sub = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(node);
    if (!sub) {
        return false;
    }

    if (ov::is_dequantization_node(sub)) {
        return false;
    }

    if (!sub->get_input_element_type(1).is_signed()) {
        return false;
    }

    std::shared_ptr<ngraph::Node> neg = std::make_shared<ngraph::opset1::Multiply>(
        sub->get_input_node_shared_ptr(1),
        opset1::Constant::create(sub->get_input_element_type(1), Shape{}, {-1}));
    if (auto constant = ov::get_constant_from_source(neg))
        neg = constant;

    auto add = std::make_shared<ngraph::opset1::Add>(sub->get_input_node_shared_ptr(0), neg);

    add->set_friendly_name(sub->get_friendly_name());
    ngraph::copy_runtime_info(sub, {neg, add});
    ngraph::replace_node(sub, add);

    return true;
}

ngraph::pass::ConvertSubtract::ConvertSubtract() {
    MATCHER_SCOPE(ConvertSubtract);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertSubtractWithConstant::ConvertSubtractWithConstant() {
    MATCHER_SCOPE(ConvertSubtractWithConstant);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>(
        {pattern::any_input(), pattern::wrap_type<op::Constant>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}
