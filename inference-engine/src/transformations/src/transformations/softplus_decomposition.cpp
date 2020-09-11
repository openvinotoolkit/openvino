// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/softplus_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::SoftPlusDecomposition, "SoftPlusDecomposition", 0);

ngraph::pass::SoftPlusDecomposition::SoftPlusDecomposition() {
    // decomposes SoftPlus(x) operation into ln(exp(x) + 1.0)
    auto input = ngraph::pattern::any_input();
    auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(input);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto softplus_input = pattern_to_output.at(input);
        auto softplus_node = pattern_to_output.at(softplus).get_node_shared_ptr();

        if (m_transformation_callback(softplus_node)) {
            return false;
        }

        auto exp = std::make_shared<ngraph::opset4::Exp>(softplus_input);
        auto add = std::make_shared<ngraph::opset4::Add>(exp,
            opset4::Constant::create(softplus_input.get_element_type(), ngraph::Shape{1}, {1.0}));
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        log->set_friendly_name(softplus_node->get_friendly_name());
        ngraph::copy_runtime_info(softplus_node, {exp, add, log});
        ngraph::replace_node(softplus_node, log);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(softplus, "SoftPlusDecomposition");
    register_matcher(m, callback);
}
