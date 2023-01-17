// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_div_with_eps_to_keep_in_mixed_precision.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "itt.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace ov;
using namespace std;

ov::pass::MarkDivWithEpsToKeepInMixedPrecision::MarkDivWithEpsToKeepInMixedPrecision() {
    MATCHER_SCOPE(MarkDivWithEpsToKeepInMixedPrecision);

    // to detect the following patterns where eps is used to prevent division by zero:
    // input_1/Maximum(input_2, eps)
    // input_1/Add(input_2, eps)
    // input_1/Sqrt(Maximum(input_2, eps))
    // input_1/Sqrt(Add(input_2, eps))
    // input_1*Pow(Maximum(input_2, eps), -z)
    // input_1*Pow(Add(input_2, eps), -z)
    auto input_1 = pattern::any_input();
    auto input_2 = pattern::any_input();

    auto eps_const_pattern = pattern::wrap_type<opset8::Constant>();
    auto max = std::make_shared<opset8::Maximum>(input_2, eps_const_pattern);
    auto add = std::make_shared<opset8::Add>(input_2, eps_const_pattern);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});

    auto sqrt = std::make_shared<opset8::Sqrt>(max_or_add);
    auto sqrt_or_max_add = std::make_shared<pattern::op::Or>(OutputVector{max_or_add, sqrt});
    // whether is divided directly or after sqrt (e.g. in L2Norm after sqrt, in MVN is divided directly)
    auto divide = std::make_shared<opset8::Divide>(input_1, sqrt_or_max_add);

    auto pow_exp = pattern::wrap_type<opset8::Constant>();
    auto convert_pattern = pattern::wrap_type<opset8::Convert>({pow_exp});
    auto pow_exp_or_convert = std::make_shared<pattern::op::Or>(OutputVector{pow_exp, convert_pattern});

    auto pow_pattern = std::make_shared<opset8::Power>(max_or_add, pow_exp_or_convert);
    auto mul_pattern = std::make_shared<opset8::Multiply>(input_1, pow_pattern);
    auto div_or_mul_to_negative_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, mul_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        if (!m.get_match_root())
            return false;

        const auto mul = std::dynamic_pointer_cast<opset8::Multiply>(m.get_match_root());
        // if pattern input_1*Pow(Maximum(input_2, eps), z) or input_1*Pow(Add(input_2, eps), z) is matched
        // need to check that power is negative
        if (mul) {
            const auto pow_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(pow_exp));
            if (pow_const) {
                // continue only if exponent is negative (z < 0)
                if (pow_const->get_element_type() == element::f16) {
                    for (auto val : pow_const->get_vector<ov::float16>())
                        if (val >= 0.0f)
                            return false;
                } else if (pow_const->get_element_type() == element::f32) {
                    for (auto val : pow_const->get_vector<float>())
                        if (val >= 0.0f)
                            return false;
                }
            }
        }

        const auto eps_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(eps_const_pattern));
        if (!eps_const)
            return false;
        if (eps_const->get_element_type() == element::f32) {
            for (const auto& val : eps_const->get_vector<float>())
                if (val > static_cast<float>(ov::float16::from_bits(0x0400)))
                    return false;
        } else if (eps_const->get_element_type() == element::f16) {
            for (const auto& val : eps_const->get_vector<ov::float16>())
                if (val > ov::float16::from_bits(0x0400))
                    return false;
        }
        disable_fp16_compression(m.get_match_root());
        return true;
    };

    auto m = make_shared<pattern::Matcher>(div_or_mul_to_negative_pow, matcher_name);
    register_matcher(m, callback);
}
