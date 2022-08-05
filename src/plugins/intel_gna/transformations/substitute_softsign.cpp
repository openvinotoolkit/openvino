// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/substitute_softsign.hpp"

#include "transformations/utils/transformation_helper.hpp"
#include "transformations/utils/utils.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <ops/softsign.hpp>

using namespace GNAPluginNS;

using Node = std::shared_ptr<ngraph::Node>;

namespace {

void DoTransformation(Node start_node, Node last_node) {
    auto activation = std::make_shared<ov::intel_gna::op::SoftSign>(start_node);
    activation->set_friendly_name(last_node->get_friendly_name());
    ngraph::copy_runtime_info(last_node, activation);
    ngraph::replace_node(last_node, activation);
}

class IsConstValueAcceptable {
public:
    IsConstValueAcceptable(double expected_value) :
        m_expected_value(expected_value) {}

    bool operator()(const ngraph::Output<ngraph::Node>& output) const {
        auto node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(output.get_node_shared_ptr());
        if (!node)
            return false;

        float value;
        if (!ngraph::op::util::get_single_value(node, value)) {
            return false;
        }

        return (value == m_expected_value);
    }

private:
    const double m_expected_value;
};

} // namespace

SubstituteSoftsign::SubstituteSoftsign() {
    MATCHER_SCOPE(SubstituteSoftsign);

    auto root = ngraph::pattern::any_input();
    auto abs = ngraph::pattern::wrap_type<ngraph::opset8::Abs>({root});

    auto add_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(IsConstValueAcceptable(1.0));
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({abs, add_const});

    auto power_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(IsConstValueAcceptable(-1.0));
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
        if (last_node_it == pattern_map.end())
            return false;
        auto last_node = last_node_it->second.get_node_shared_ptr();

        DoTransformation(root_node, last_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(last, matcher_name);
    this->register_matcher(m, callback);
}
