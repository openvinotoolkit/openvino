// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/softplus_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"

ngraph::pass::SoftPlusFusion::SoftPlusFusion() {
    MATCHER_SCOPE(SoftPlusFusion);
    // fuses ln(exp(x) + 1.0) operations into SoftPlus(x)
    auto input = ngraph::pattern::any_input();
    auto exp = std::make_shared<ngraph::opset4::Exp>(input);
    auto add_constant =
        ngraph::pattern::wrap_type<ngraph::opset4::Constant>(pattern::type_matches_any({element::f32, element::f16}));
    auto add = std::make_shared<ngraph::opset4::Add>(exp, add_constant);
    auto log = std::make_shared<ngraph::opset4::Log>(add);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = std::dynamic_pointer_cast<ngraph::opset4::Constant>(
            pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!constant)
            return false;

        auto data = constant->cast_vector<float>();
        if (data.size() != 1 || data[0] != 1.0) {
            return false;
        }

        auto softplus = std::make_shared<ngraph::opset4::SoftPlus>(exp_input);

        softplus->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(log).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr()},
                                  softplus);
        ngraph::replace_node(m.get_match_root(), softplus);
        MATCHER_SCOPE_ENABLE(SoftPlusFusion);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(log, matcher_name);
    register_matcher(m, callback);
}
