// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/hswish_fusion.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

bool check_constant_value(const std::shared_ptr<ngraph::opset4::Constant>& constant, const float value) {
    if (!constant) {
        return false;
    }
    if (constant->get_element_type() == ngraph::element::f32 || constant->get_element_type() == ngraph::element::f16) {
        auto data = constant->cast_vector<float>();
        if (data.size() != 1 || data[0] != value) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

ngraph::pass::HSwishFusionWithRelu::HSwishFusionWithRelu() {
    // replaces a sub-graphs x * (min(Relu(x + 3), 6) / 6) with a HSwish op.
    auto input = ngraph::pattern::any_input();
    auto add_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto add = std::make_shared<ngraph::opset4::Add>(input, add_constant);
    auto relu = std::make_shared<ngraph::opset4::Relu>(add);
    auto min_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto min = std::make_shared<ngraph::opset4::Minimum>(relu, min_constant);
    auto div_constant = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto div = std::make_shared<ngraph::opset4::Divide>(min, div_constant);
    auto mul = std::make_shared<ngraph::opset4::Multiply>(input, div);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = check_constant_value(add_const_value, 3.0) && check_constant_value(min_const_value, 6.0) && 
        check_constant_value(div_const_value, 6.0);

        if (!valid_constant_values) {
            return false;
        }

        auto hswish = std::make_shared<ngraph::opset4::HSwish>(pattern_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({ 
                                    pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr(),
                                    pattern_to_output.at(relu).get_node_shared_ptr(),
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "HSwishWithReluFusion");
    register_matcher(m, callback);
}
