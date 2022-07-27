// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/division_by_zero_fp16_resolver.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <vector>

#include "itt.hpp"
#include "ngraph/rt_info.hpp"
#include "transformations/utils/utils.hpp"

constexpr float normalized_fp16_min = 6.103515625e-05f;  // fp16 minimal normalized  value

using namespace ov;

ov::pass::DivisionByZeroFP16Resolver::DivisionByZeroFP16Resolver() {
    MATCHER_SCOPE(DivisionByZeroFP16Resolver);

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
    auto pow_pattern = std::make_shared<opset8::Power>(max_or_add, pow_exp);
    auto mul_pattern = std::make_shared<opset8::Multiply>(input_1, pow_pattern);
    auto div_or_mul_to_negative_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, mul_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        const auto mul = std::dynamic_pointer_cast<opset8::Multiply>(m.get_match_root());
        if (mul) {
            // pattern input_1*Pow(Maximum(input_2, eps), z) or input_1*Pow(Add(input_2, eps), z) is matched
            const auto pow_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(pow_exp));
            for (float val : pow_const->get_vector<float>())
                if (val >= 0)  // continue only if exponent is negative (z < 0)
                    return false;
        }

        const auto eps_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(eps_const_pattern));
        if (!eps_const || eps_const->get_element_type() != ov::element::f32)
            return false;

        for (float val : eps_const->get_vector<float>())
            if (val >= normalized_fp16_min)
                return false;

        auto new_constant = std::make_shared<opset8::Constant>(eps_const->get_element_type(),
                                                               eps_const->get_shape(),
                                                               normalized_fp16_min);
        copy_runtime_info(eps_const, new_constant);
        replace_node(eps_const, new_constant);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(div_or_mul_to_negative_pow, matcher_name);
    register_matcher(m, callback);
}
