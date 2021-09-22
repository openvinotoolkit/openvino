// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusion, "HSwishFusion", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithReluDiv, "HSwishFusionWithReluDiv", 0);

ngraph::pass::HSwishFusionWithReluDiv::HSwishFusionWithReluDiv() {
    MATCHER_SCOPE(HSwishFusionWithReluDiv);
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) / 6 with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
    auto relu = std::make_shared<ngraph::opset7::Relu>(add);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
    auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0)
                                        && op::util::has_constant_value<float>(min_const_value, 6.0)
                                        && op::util::has_constant_value<float>(div_const_value, 6.0);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset7::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr(),
                                    pattern_to_output.at(relu).get_node_shared_ptr(),
                                    pattern_to_output.at(min_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(min).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr(),
                                    pattern_to_output.at(div_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(div).get_node_shared_ptr(),
                                   },
                                  hswish);
        ngraph::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithReluMul, "HSwishFusionWithReluMul", 0);

ngraph::pass::HSwishFusionWithReluMul::HSwishFusionWithReluMul() {
    MATCHER_SCOPE(HSwishFusionWithReluMul);
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
    auto relu = std::make_shared<ngraph::opset7::Relu>(add);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
    auto mul_first = std::make_shared<ngraph::opset7::Multiply>(input, min);
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto mul_second = std::make_shared<ngraph::opset7::Multiply>(mul_first, mul_constant);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset7::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        bool valid_constant_values =  op::util::has_constant_value<float>(add_const_value, 3.0f)
                                        &&  op::util::has_constant_value<float>(min_const_value, 6.0f)
                                        &&  op::util::has_constant_value<float>(mul_const_value, (1.0f/6.0f), 0.0001f);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset7::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr(),
                                    pattern_to_output.at(relu).get_node_shared_ptr(),
                                    pattern_to_output.at(min_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(min).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_first).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_second).get_node_shared_ptr()
                                   },
                                  hswish);
        ngraph::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_second, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithHSigmoid, "HSwishFusionWithHSigmoid", 0);

ngraph::pass::HSwishFusionWithHSigmoid::HSwishFusionWithHSigmoid() {
    MATCHER_SCOPE(HSwishFusionWithHSigmoid);
    // Replaces a sub-graph x * HSigmoid(x) with a HSwish op.
    auto input = pattern::any_input();
    auto hsigmoid_pattern = pattern::wrap_type<ngraph::opset7::HSigmoid>({input}, pattern::consumers_count(1));
    auto mul_pattern = pattern::wrap_type<ngraph::opset7::Multiply>({input, hsigmoid_pattern});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto hsigmoid = pattern_to_output.at(hsigmoid_pattern).get_node_shared_ptr();
        auto mul = pattern_to_output.at(mul_pattern).get_node_shared_ptr();

        auto hswish = std::make_shared<ngraph::opset7::HSwish>(pattern_to_output.at(input));
        hswish->set_friendly_name(mul->get_friendly_name());
        ngraph::copy_runtime_info({hsigmoid, mul}, hswish);
        ngraph::replace_node(mul, hswish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, matcher_name);
    register_matcher(m, callback);
}
