// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/mish_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MishFusion, "MishFusion", 0);

ngraph::pass::MishFusion::MishFusion() {
    auto input = ngraph::pattern::any_input();
    auto exp = std::make_shared<ngraph::opset4::Exp>(input);
    auto add = std::make_shared<ngraph::opset4::Add>(exp, ngraph::pattern::wrap_type<ngraph::opset4::Constant>());
    auto log = std::make_shared<ngraph::opset4::Log>(add);
    auto tanh = std::make_shared<ngraph::opset4::Tanh>(log);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, tanh);

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto mish = std::make_shared<ngraph::opset4::Mish>(exp_input);

        mish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(tanh).get_node_shared_ptr(),
                                   pattern_to_output.at(log).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr()}, mish);
        ngraph::replace_node(m.get_match_root(), mish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "MishFusion");
    register_matcher(m, matcher_pass_callback);
}
