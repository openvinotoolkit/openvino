// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/substitute_softsign.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <ops/softsign.hpp>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SubstituteSoftsign, "SubstituteSoftsign", 0);

using Node = std::shared_ptr<ngraph::Node>;

namespace {

void DoTransformation(Node start_node, Node last_node) {
    auto activation = std::make_shared<SoftSign>(start_node);
    activation->set_friendly_name(last_node->get_friendly_name());
    ngraph::copy_runtime_info(last_node, activation);
    ngraph::replace_node(last_node, activation);
}

bool IsAddConstAcceptable(const ngraph::Output<ngraph::Node>& output) {
    auto node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(output.get_node_shared_ptr());
    if (!node)
        return false;

    const std::vector<double> & values = node->cast_vector<double>();

    return (std::find_if_not(values.begin(), values.end(), [](double d) { return d == 1.0; }) == values.end());
}

bool IsPowerConstAcceptable(const ngraph::Output<ngraph::Node>& output) {
    auto node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(output.get_node_shared_ptr());
    if (!node)
        return false;

    const std::vector<double> & values = node->cast_vector<double>();

    return (std::find_if_not(values.begin(), values.end(), [](double d) { return d == -1.0; }) == values.end());
}

} // namespace

SubstituteSoftsign::SubstituteSoftsign() {
    MATCHER_SCOPE(SubstituteSoftsign);

    auto root = ngraph::pattern::any_input();
    auto abs = ngraph::pattern::wrap_type<ngraph::opset8::Abs>({root});

    auto add_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(IsAddConstAcceptable);
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({abs, add_const});

    auto power_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(IsPowerConstAcceptable);
    auto power = ngraph::pattern::wrap_type<ngraph::opset8::Power>({add, power_const});

    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({root, power});
    auto divide = ngraph::pattern::wrap_type<ngraph::opset8::Divide>({root, add});
    auto last =  std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{multiply, divide});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root_node = pattern_map.at(root).get_node_shared_ptr();

        auto last_node_it = pattern_map.find(multiply);
        if (last_node_it == pattern_map.end())
            last_node_it = pattern_map.find(divide);
        auto last_node = last_node_it->second.get_node_shared_ptr();

        DoTransformation(root_node, last_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(last, matcher_name);
    this->register_matcher(m, callback);
}
