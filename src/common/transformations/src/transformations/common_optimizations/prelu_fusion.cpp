// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "transformations/common_optimizations/prelu_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::PReluFusionNegativeAdd::PReluFusionNegativeAdd() {
    MATCHER_SCOPE(PReluFusionNegativeAdd);
    auto input = pass::pattern::any_input();
    auto relu_pos = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({input});
    auto neg1 = ov::pass::pattern::wrap_type<ov::op::v0::Negative>({input});
    auto relu_neg = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({neg1});
    auto neg2 = ov::pass::pattern::wrap_type<ov::op::v0::Negative>({relu_neg});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({neg2, mul_constant});
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({relu_pos, mul});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        auto prelu = std::make_shared<ov::op::v0::PRelu>(input_output, slope_output);
        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                    pattern_to_output.at(neg1).get_node_shared_ptr(),
                                    pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(neg2).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr()};
        ov::copy_runtime_info(copy_from, prelu);
        ov::replace_node(add_node, prelu);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PReluFusionNegativeSub::PReluFusionNegativeSub() {
    MATCHER_SCOPE(PReluFusionNegativeSub);
    auto input = pass::pattern::any_input();
    auto relu_pos = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({input});
    auto neg1 = ov::pass::pattern::wrap_type<ov::op::v0::Negative>({input});
    auto relu_neg = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({neg1});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({relu_neg, mul_constant});
    auto sub = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({relu_pos, mul});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto sub_node = pattern_to_output.at(sub).get_node_shared_ptr();
        auto prelu = std::make_shared<ov::op::v0::PRelu>(input_output, slope_output);
        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                    pattern_to_output.at(neg1).get_node_shared_ptr(),
                                    pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr(),
                                    pattern_to_output.at(sub).get_node_shared_ptr()};
        ov::copy_runtime_info(copy_from, prelu);
        ov::replace_node(sub_node, prelu);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(sub, matcher_name);
    register_matcher(m, callback);
}

static std::function<bool(ov::Output<ov::Node>)> constant_value(const float target_value) {
    return [=](const ov::Output<ov::Node>& output) -> bool {
        auto node = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
        if (!node) {
            return false;
        }
        float value;
        if (!ov::op::util::get_single_value(node, value)) {
            return false;
        }
        return value == target_value;
    };
}

ov::pass::PReluFusionMultiplyAdd::PReluFusionMultiplyAdd() {
    MATCHER_SCOPE(PReluFusionMultiplyAdd);
    auto input = pass::pattern::any_input();
    auto relu_pos = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({input});
    auto mul_neg_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(constant_value(-1.0));
    auto mul_neg = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_neg_constant});
    auto relu_neg = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({mul_neg});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({relu_neg, mul_constant});
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({relu_pos, mul});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        auto negative = ov::op::util::make_try_fold<ov::op::v0::Negative>(slope_output);
        auto prelu = std::make_shared<ov::op::v0::PRelu>(input_output, negative);

        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr(),
                                    pattern_to_output.at(add).get_node_shared_ptr()};
        ov::copy_runtime_info(copy_from, {prelu, negative});
        ov::replace_node(add_node, prelu);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PReluFusionMultiplySub::PReluFusionMultiplySub() {
    MATCHER_SCOPE(PReluFusionMultiplySub);
    auto input = pass::pattern::any_input();
    auto relu_pos = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({input});
    auto mul_neg_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(constant_value(-1.0));
    auto mul_neg = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, mul_neg_constant});
    auto relu_neg = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({mul_neg});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({relu_neg, mul_constant});
    auto sub = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({relu_pos, mul});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto input_output = pattern_to_output.at(input);
        auto slope_output = pattern_to_output.at(mul_constant);
        auto sub_node = pattern_to_output.at(sub).get_node_shared_ptr();
        auto prelu = std::make_shared<ov::op::v0::PRelu>(input_output, slope_output);

        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                    pattern_to_output.at(mul_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                    pattern_to_output.at(mul).get_node_shared_ptr(),
                                    pattern_to_output.at(sub).get_node_shared_ptr()};
        ov::copy_runtime_info(copy_from, prelu);
        ov::replace_node(sub_node, prelu);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(sub, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PReluFusionAbsSubMulMulAdd::PReluFusionAbsSubMulMulAdd() {
    MATCHER_SCOPE(PReluFusionAbsSubMulMulAdd);

    using namespace std;
    using namespace ov;

    const auto equals_half = [](const Output<Node>& node) {
        float v;
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node.get_node_shared_ptr());
        return constant && op::util::get_single_value(constant, v) && v == 0.5f;
    };

    const auto input = pass::pattern::any_input();
    const auto relu = pattern::wrap_type<ov::op::v0::Relu>({input});
    const auto abs = pattern::wrap_type<ov::op::v0::Abs>({input});
    const auto sub = pattern::wrap_type<ov::op::v1::Subtract>({input, abs});
    const auto mul_1_constant = pattern::wrap_type<ov::op::v0::Constant>();
    const auto mul_1 = pattern::wrap_type<ov::op::v1::Multiply>({sub, mul_1_constant});
    const auto mul_2_constant = pattern::wrap_type<ov::op::v0::Constant>(equals_half);
    const auto mul_2 = pattern::wrap_type<ov::op::v1::Multiply>({mul_1, mul_2_constant});
    const auto add = pattern::wrap_type<ov::op::v1::Add>({mul_2, relu});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto input_output = pattern_to_output.at(input);
        const auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        const auto slope = pattern_to_output.at(mul_1_constant);
        const auto prelu = make_shared<ov::op::v0::PRelu>(input_output, slope);

        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        const OutputVector copy_from = {pattern_to_output.at(relu),
                                        pattern_to_output.at(abs),
                                        pattern_to_output.at(sub),
                                        pattern_to_output.at(mul_1),
                                        pattern_to_output.at(mul_2),
                                        pattern_to_output.at(add)};
        copy_runtime_info(as_node_vector(copy_from), prelu);
        replace_node(add_node, prelu);
        return true;
    };
    auto m = make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PReluFusionNegReluMulAdd::PReluFusionNegReluMulAdd() {
    MATCHER_SCOPE(PReluFusionNegReluMulAdd);

    using namespace std;
    using namespace ov;

    const auto input = pass::pattern::any_input();
    const auto relu_pos = pattern::wrap_type<ov::op::v0::Relu>({input});
    const auto neg1 = pattern::wrap_type<ov::op::v0::Negative>({input});
    const auto relu_neg = pattern::wrap_type<ov::op::v0::Relu>({neg1});
    const auto mul_constant = pattern::wrap_type<ov::op::v0::Constant>();
    const auto mul = pattern::wrap_type<ov::op::v1::Multiply>({relu_neg, mul_constant});
    const auto add = pattern::wrap_type<ov::op::v1::Add>({relu_pos, mul});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto input_output = pattern_to_output.at(input);
        const auto add_node = pattern_to_output.at(add).get_node_shared_ptr();
        const auto slope = op::util::make_try_fold<ov::op::v0::Negative>(pattern_to_output.at(mul_constant));
        const auto prelu = make_shared<ov::op::v0::PRelu>(input_output, slope);
        prelu->set_friendly_name(m.get_match_root()->get_friendly_name());
        NodeVector copy_from = {pattern_to_output.at(relu_pos).get_node_shared_ptr(),
                                pattern_to_output.at(neg1).get_node_shared_ptr(),
                                pattern_to_output.at(relu_neg).get_node_shared_ptr(),
                                pattern_to_output.at(mul).get_node_shared_ptr(),
                                pattern_to_output.at(add).get_node_shared_ptr()};
        copy_runtime_info(copy_from, prelu);
        replace_node(add_node, prelu);
        return true;
    };
    auto matcher = make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(matcher, callback);
}
