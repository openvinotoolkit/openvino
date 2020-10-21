// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusion, "HSwishFusion", 0);

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithReluDiv, "HSwishFusionWithReluDiv", 0);

ngraph::pass::HSwishFusionWithReluDiv::HSwishFusionWithReluDiv() {
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) / 6 with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
    auto relu = std::make_shared<ngraph::opset4::Relu>(add);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto min = std::make_shared<ngraph::opset4::Minimum>(relu, min_constant);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, min);
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto div = std::make_shared<ngraph::opset4::Divide>(mul, div_constant);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0)
                                        && op::util::has_constant_value<float>(min_const_value, 6.0)
                                        && op::util::has_constant_value<float>(div_const_value, 6.0);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset4::HSwish>(x_output);

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

    auto m = std::make_shared<ngraph::pattern::Matcher>(div, "HSwishWithReluDivFusion");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithReluMul, "HSwishFusionWithReluMul", 0);

ngraph::pass::HSwishFusionWithReluMul::HSwishFusionWithReluMul() {
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
    auto relu = std::make_shared<ngraph::opset4::Relu>(add);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto min = std::make_shared<ngraph::opset4::Minimum>(relu, min_constant);
    auto mul_first = std::make_shared<ngraph::opset4::Multiply>(input, min);
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto mul_second = std::make_shared<ngraph::opset4::Multiply>(mul_first, mul_constant);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        bool valid_constant_values =  op::util::has_constant_value<float>(add_const_value, 3.0f)
                                        &&  op::util::has_constant_value<float>(min_const_value, 6.0f)
                                        &&  op::util::has_constant_value<float>(mul_const_value, (1.0f/6.0f), 0.0001f);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset4::HSwish>(x_output);

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

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_second, "HSwishWithReluMulFusion");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithoutRelu, "HSwishFusionWithoutRelu", 0);

ngraph::pass::HSwishFusionWithoutRelu::HSwishFusionWithoutRelu() {
    // Replaces a sub-graph x * (min(max(x + 3, 0), 6) / 6) with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
    auto max_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto max = std::make_shared<ngraph::opset4::Maximum>(add, max_constant);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto min = std::make_shared<ngraph::opset4::Minimum>(max, min_constant);
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto div = std::make_shared<ngraph::opset4::Divide>(min, div_constant);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, div);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto max_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(max_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0f)
                                        && op::util::has_constant_value<float>(max_const_value, 0.0f)
                                        && op::util::has_constant_value<float>(min_const_value, 6.0f)
                                        && op::util::has_constant_value<float>(div_const_value, 6.0f);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset4::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr(),
                                    pattern_to_output.at(max_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(max).get_node_shared_ptr(),
                                    pattern_to_output.at(min_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(min).get_node_shared_ptr(),
                                    pattern_to_output.at(div_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(div).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr()
                                   },
                                  hswish);
        ngraph::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "HSwishWithoutReluFusion");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::HSwishFusionWithClamp, "HSwishFusionWithClamp", 0);

ngraph::pass::HSwishFusionWithClamp::HSwishFusionWithClamp() {
    // Replaces a sub-graph x * (Clamp(x + 3, 0, 6) * const(1/6)) with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
    auto clamp = std::make_shared<ngraph::op::v0::Clamp>(add, 0.0f, 6.0f);
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto mul_first = std::make_shared<ngraph::opset4::Multiply>(clamp, mul_constant);
    auto mul_second = std::make_shared<ngraph::opset4::Multiply>(input, mul_first);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value(add_const_value, 3.0)
                                     && op::util::has_constant_value(mul_const_value, (1.0/6.0), 0.0001);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset4::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr(),
                                    pattern_to_output.at(clamp).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_first).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_second).get_node_shared_ptr()
                                  },
                                  hswish);
        ngraph::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_second, "HSwishWithClampFusion");
    register_matcher(m, callback);
}
