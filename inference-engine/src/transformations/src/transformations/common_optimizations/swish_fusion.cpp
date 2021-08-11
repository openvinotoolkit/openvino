// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/swish_fusion.hpp"

#include <memory>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

bool check_beta_value(const std::shared_ptr<ov::opset4::Constant>& constant) {
    // check that the constant for beta contains only one distinct element
    if (!constant) {
        return false;
    }
    if (constant->get_element_type() == ov::element::f32 || constant->get_element_type() == ov::element::f16) {
        auto data = constant->cast_vector<float>();
        if (!std::equal(data.begin() + 1, data.end(), data.begin())) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

NGRAPH_RTTI_DEFINITION(ov::pass::SwishFusion, "SwishFusion", 0);

NGRAPH_RTTI_DEFINITION(ov::pass::SwishFusionWithSigmoid, "SwishFusionWithSigmoid", 0);

ov::pass::SwishFusionWithSigmoid::SwishFusionWithSigmoid() {
    MATCHER_SCOPE(SwishFusionWithSigmoid);
    // replaces a sub-graphs x * Sigmoid(x) with a Swish op.
    auto input = ov::pattern::any_input();
    auto sigmoid = std::make_shared<ov::opset4::Sigmoid>(input);
    auto mul = std::make_shared<ov::opset4::Multiply>(input, sigmoid);

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto swish = std::make_shared<ov::opset4::Swish>(exp_input);

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(sigmoid).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr()},
                                  swish);
        ov::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::SwishFusionWithSigmoidWithBeta, "SwishFusionWithSigmoidWithBeta", 0);

ov::pass::SwishFusionWithSigmoidWithBeta::SwishFusionWithSigmoidWithBeta() {
    MATCHER_SCOPE(SwishFusionWithSigmoidWithBeta);
    // replaces a sub-graphs x * Sigmoid(x * beta) with a Swish op.
    auto input = ov::pattern::any_input();
    auto beta = ov::pattern::any_input();
    auto mul_beta = std::make_shared<ov::opset4::Multiply>(input, beta);
    auto sigmoid = std::make_shared<ov::opset4::Sigmoid>(mul_beta);
    auto mul = std::make_shared<ov::opset4::Multiply>(input, sigmoid);

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);
        auto beta_input = pattern_to_output.at(beta);

        auto beta_constant = std::dynamic_pointer_cast<ov::opset4::Constant>(beta_input.get_node_shared_ptr());
        Output<Node> new_beta;
        if (beta_constant) {
            if (check_beta_value(beta_constant)) {
                new_beta = opset4::Constant::create(beta_input.get_element_type(), Shape{}, {beta_constant->cast_vector<float>()[0]});
            } else {
                return false;
            }
        } else {
            // if the input is not constant and number of elements is not equal to 1 then we cannot perform fusing
            if (beta_input.get_partial_shape().is_dynamic() || ov::shape_size(beta_input.get_shape()) != 1) {
                return false;
            }
            new_beta = beta_input;
        }

        auto swish = std::make_shared<ov::opset4::Swish>(exp_input, new_beta);

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(sigmoid).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr()},
                                  swish);
        ov::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::SwishFusionWithBeta, "SwishFusionWithBeta", 0);

ov::pass::SwishFusionWithBeta::SwishFusionWithBeta() {
    MATCHER_SCOPE(SwishFusionWithBeta);
    // replaces a sub-graphs x / (1.0 + exp(-x * beta)) with a Swish op.
    auto input = ov::pattern::any_input();
    auto beta = ov::pattern::any_input();
    auto mul = std::make_shared<ov::opset4::Multiply>(input, beta);
    auto neg = std::make_shared<ov::opset4::Negative>(mul);
    auto exp = std::make_shared<ov::opset4::Exp>(neg);
    auto add_constant = ov::pattern::wrap_type<ov::opset4::Constant>();
    auto add = std::make_shared<ov::opset4::Add>(exp, add_constant);
    auto div = std::make_shared<ov::opset4::Divide>(input, add);

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher &m) {
        auto &pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = std::dynamic_pointer_cast<ov::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!op::util::has_constant_value<float>(constant, 1.0f)) {
            return false;
        }

        auto swish = std::make_shared<ov::opset4::Swish>(exp_input, pattern_to_output.at(beta));

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(beta).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(neg).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr(),
                                   pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(div).get_node_shared_ptr()},
                                  swish);
        ov::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::SwishFusionWithoutBeta, "SwishFusionWithoutBeta", 0);

ov::pass::SwishFusionWithoutBeta::SwishFusionWithoutBeta() {
    MATCHER_SCOPE(SwishFusionWithoutBeta);
    // replaces a sub-graphs x / (1.0 + exp(-x)) with a Swish op.
    auto input = ov::pattern::any_input();
    auto neg = std::make_shared<ov::opset4::Negative>(input);
    auto exp = std::make_shared<ov::opset4::Exp>(neg);
    auto add_constant = ov::pattern::wrap_type<ov::opset4::Constant>();
    auto add = std::make_shared<ov::opset4::Add>(exp, add_constant);
    auto div = std::make_shared<ov::opset4::Divide>(input, add);

    ov::matcher_pass_callback callback = [=](ov::pattern::Matcher& m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(input);

        auto constant = std::dynamic_pointer_cast<ov::opset4::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!op::util::has_constant_value<float>(constant, 1.0f)) {
            return false;
        }

        auto swish = std::make_shared<ov::opset4::Swish>(exp_input);

        swish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(neg).get_node_shared_ptr(),
                                   pattern_to_output.at(exp).get_node_shared_ptr(),
                                   pattern_to_output.at(add_constant).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(div).get_node_shared_ptr()},
                                   swish);
        ov::replace_node(m.get_match_root(), swish);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}
