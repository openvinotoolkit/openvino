// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/softplus_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::SoftPlusDecomposition::SoftPlusDecomposition() {
    auto input = ngraph::pattern::any_input();
    auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(input);

    ngraph::graph_rewrite_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto softplus_input = pattern_to_output.at(input);

        auto softplus_node = std::dynamic_pointer_cast<ngraph::opset4::SoftPlus>(pattern_to_output.at(softplus).get_node_shared_ptr());

        if (m_transformation_callback(softplus_node)) {
            return false;
        }

        auto exp = std::make_shared<ngraph::opset4::Exp>(softplus_input);
        auto add = std::make_shared<ngraph::opset4::Add>(exp,
            opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0}));
        auto log = std::make_shared<ngraph::opset4::Log>(add);

        log->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(pattern_to_output.at(softplus).get_node_shared_ptr(), {exp, add, log});
        ngraph::replace_node(m.get_match_root(), log);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(softplus, "SoftPlusDecomposition");
    register_matcher(m, callback);
}
