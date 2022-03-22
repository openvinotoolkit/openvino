// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/prelu_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::PReluFusion, "PReluFusion", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::PReluFusionNegativeAdd, "PReluFusionNegativeAdd", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::PReluFusionNegativeSub, "PReluFusionNegativeSub", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::PReluFusionMultiplyAdd, "PReluFusionMultiplyAdd", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::PReluFusionMultiplySub, "PReluFusionMultiplySub", 0);

ngraph::pass::PReluFusionNegativeAdd::PReluFusionNegativeAdd() {
    MATCHER_SCOPE(PReluFusionNegativeAdd);
    auto input = ngraph::pattern::any_input();
    auto relu_pos = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({input});
    auto neg1 = ngraph::pattern::wrap_type<ngraph::opset8::Negative>({input});
    auto relu_neg = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({neg1});
    auto neg2 = ngraph::pattern::wrap_type<ngraph::opset8::Negative>({relu_neg});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({neg2, mul_constant});
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({relu_pos, mul});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        auto prelu = std::make_shared<ngraph::opset8::PRelu>(input_output, slope_output);
        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                        pattern_to_output.at(neg1).get_node_shared_ptr(),
                                        pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(neg2).get_node_shared_ptr(),
                                        pattern_to_output.at(mul).get_node_shared_ptr(),
                                        pattern_to_output.at(add).get_node_shared_ptr()};
        ngraph::copy_runtime_info(copy_from, prelu);
        ngraph::replace_node(add_node, prelu);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::PReluFusionNegativeSub::PReluFusionNegativeSub() {
    MATCHER_SCOPE(PReluFusionNegativeSub);
    auto input = ngraph::pattern::any_input();
    auto relu_pos = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({input});
    auto neg1 = ngraph::pattern::wrap_type<ngraph::opset8::Negative>({input});
    auto relu_neg = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({neg1});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({relu_neg, mul_constant});
    auto sub = ngraph::pattern::wrap_type<ngraph::opset8::Subtract>({relu_pos, mul});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto sub_node = pattern_to_output.at(sub).get_node_shared_ptr();
        auto prelu = std::make_shared<ngraph::opset8::PRelu>(input_output, slope_output);
        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                        pattern_to_output.at(neg1).get_node_shared_ptr(),
                                        pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(mul).get_node_shared_ptr(),
                                        pattern_to_output.at(sub).get_node_shared_ptr()};
        ngraph::copy_runtime_info(copy_from, prelu);
        ngraph::replace_node(sub_node, prelu);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    register_matcher(m, callback);
}

static std::function<bool(ngraph::Output<ngraph::Node>)> constant_value(const float target_value) {
    return [=](const ngraph::Output<ngraph::Node>& output) -> bool {
        auto node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(output.get_node_shared_ptr());
        if (!node) {
            return false;
        }
        float value;
        if (!ngraph::op::util::get_single_value(node, value)) {
            return false;
        }
        return value == target_value;
    };
}

ngraph::pass::PReluFusionMultiplyAdd::PReluFusionMultiplyAdd() {
    MATCHER_SCOPE(PReluFusionMultiplyAdd);
    auto input = ngraph::pattern::any_input();
    auto relu_pos = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({input});
    auto mul_neg_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(constant_value(-1.0));
    auto mul_neg = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({input, mul_neg_constant});
    auto relu_neg = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({mul_neg});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({relu_neg, mul_constant});
    auto add = ngraph::pattern::wrap_type<ngraph::opset8::Add>({relu_pos, mul});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        auto negative = ngraph::op::util::make_try_fold<ngraph::opset8::Negative>(slope_output);
        auto prelu = std::make_shared<ngraph::opset8::PRelu>(input_output, negative);

        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                        pattern_to_output.at(mul_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(mul).get_node_shared_ptr(),
                                        pattern_to_output.at(add).get_node_shared_ptr()};
        ngraph::copy_runtime_info(copy_from, {prelu, negative});
        ngraph::replace_node(add_node, prelu);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ngraph::pass::PReluFusionMultiplySub::PReluFusionMultiplySub() {
    MATCHER_SCOPE(PReluFusionMultiplySub);
    auto input = ngraph::pattern::any_input();
    auto relu_pos = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({input});
    auto mul_neg_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>(constant_value(-1.0));
    auto mul_neg = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({input, mul_neg_constant});
    auto relu_neg = ngraph::pattern::wrap_type<ngraph::opset8::Relu>({mul_neg});
    auto mul_constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = ngraph::pattern::wrap_type<ngraph::opset8::Multiply>({relu_neg, mul_constant});
    auto sub = ngraph::pattern::wrap_type<ngraph::opset8::Subtract>({relu_pos, mul});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto sub_node = pattern_to_output.at(sub).get_node_shared_ptr();
        auto prelu = std::make_shared<ngraph::opset8::PRelu>(input_output, slope_output);

        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                        pattern_to_output.at(mul_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                        pattern_to_output.at(mul).get_node_shared_ptr(),
                                        pattern_to_output.at(sub).get_node_shared_ptr()};
        ngraph::copy_runtime_info(copy_from, prelu);
        ngraph::replace_node(sub_node, prelu);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    register_matcher(m, callback);
}