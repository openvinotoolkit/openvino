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

using namespace ngraph;

static bool convert_subtract(const std::shared_ptr<Node>& node) {
    auto sub = std::dynamic_pointer_cast<opset1::Subtract>(node);
    if (!sub) {
        return false;
    }

    if (ov::is_dequantization_node(sub)) {
        return false;
    }

    if (!sub->get_input_element_type(1).is_signed()) {
        return false;
    }

    std::shared_ptr<Node> neg = std::make_shared<opset1::Multiply>(
        sub->get_input_node_shared_ptr(1),
        opset1::Constant::create(sub->get_input_element_type(1), Shape{}, {-1}));
    NodeVector new_nodes;
    if (auto constant = ov::get_constant_from_source(neg)) {
        neg = constant;
    } else {
        new_nodes.push_back(neg);
    }

    auto add = std::make_shared<opset1::Add>(sub->get_input_node_shared_ptr(0), neg);
    new_nodes.push_back(add);

    add->set_friendly_name(sub->get_friendly_name());
    copy_runtime_info(sub, new_nodes);
    replace_node(sub, add);

    return true;
}

pass::ConvertSubtract::ConvertSubtract() {
    MATCHER_SCOPE(ConvertSubtract);
    auto sub = pattern::wrap_type<opset1::Subtract>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}

pass::ConvertSubtractWithConstant::ConvertSubtractWithConstant() {
    MATCHER_SCOPE(ConvertSubtractWithConstant);
    auto sub = pattern::wrap_type<opset1::Subtract>(
        {pattern::any_input(), pattern::wrap_type<op::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        return convert_subtract(node);
    };

    auto m = std::make_shared<pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}
