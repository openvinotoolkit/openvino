// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/division_by_zero_fp16_resolver.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <openvino/opsets/opset8.hpp>
#include "ngraph/rt_info.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::DivisionByZeroFP16Resolver, "DivisionByZeroFP16Resolver", 0);

constexpr float normalized_fp16_min = 6.103515625e-05f;  // fp16 minimal normalized  value

using namespace ov;

ov::pass::DivisionByZeroFP16Resolver::DivisionByZeroFP16Resolver() {
    MATCHER_SCOPE(DivisionByZeroFP16Resolver);
    auto input_1 = pattern::any_input();
    auto input_2 = pattern::any_input();

    auto eps_const_pattern = pattern::wrap_type<opset8::Constant>();
    auto max = std::make_shared<opset8::Maximum>(input_2, eps_const_pattern);
    auto add = std::make_shared<opset8::Add>(input_2, eps_const_pattern);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});
    auto divide = std::make_shared<opset8::Divide>(input_1, max_or_add);
    auto pow_exp = pattern::wrap_type<opset8::Constant>();
    auto pow_pattern = std::make_shared<opset8::Power>(max_or_add, pow_exp);
    auto div_or_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, pow_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto pow = std::dynamic_pointer_cast<opset8::Power>(m.get_match_root());
        if (pow) {
            const auto pow_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(pow_exp).get_node_shared_ptr());
            for (float val : pow_const->get_vector<float>())
                if (val >= 0)  // only for negative exponents
                    return false;
        }

        const auto eps_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_to_output.at(eps_const_pattern).get_node_shared_ptr());
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

    auto m = std::make_shared<pattern::Matcher>(div_or_pow, matcher_name);
    register_matcher(m, callback);
}
