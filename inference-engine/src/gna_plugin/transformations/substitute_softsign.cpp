// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/substitute_softsign.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SubstituteSoftsign, "SubstituteSoftsign", 0);

using Node = std::shared_ptr<ngraph::Node>;

namespace {

// TODO

void DoTransformation(Node start_node, Node last_node)
{
    // TODO
}

} // namespace

SubstituteSoftsign::SubstituteSoftsign() {
    MATCHER_SCOPE(SubstituteSoftsign);

    auto root = ngraph::pattern::any_input();
    auto abs = ngraph::pattern::wrap_type<ngraph::opset8::Abs>({root});

    auto add_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(); /* TODO: check const the same from MO */
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({abs, add_const});

    auto add_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{abs, add});

    auto power_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(); /* TODO: check const = -1 */
    auto power = ngraph::pattern::wrap_type<ngraph::opset8::Power>({add_output, power_const});

    auto multiply = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({root, power});
    auto divide = ngraph::pattern::wrap_type<ngraph::opset8::Divide>({root, add_output});
    auto last =  std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{multiply, divide});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root_node = pattern_map.at(root).get_node_shared_ptr();
        auto last_node = pattern_map.at(last).get_node_shared_ptr();

        DoTransformation(root_node, last);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(last, matcher_name);
    this->register_matcher(m, callback);
}
