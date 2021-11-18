// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/division_to_zero_fp16_resolver.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::DivisionToZeroFP16Resolver, "DivisionToZeroFP16Resolver", 0);

constexpr float normalized_fp16_min = 6.103515625e-05f;  // normalized minimum of fp16

ngraph::pass::DivisionToZeroFP16Resolver::DivisionToZeroFP16Resolver() {
    MATCHER_SCOPE(DivisionToZeroFP16Resolver);
    auto input_1 = ngraph::pattern::any_input();
    auto input_2 = ngraph::pattern::any_input();


    auto eps_const_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto max = std::make_shared<ngraph::opset8::Maximum>(input_2, eps_const_pattern);
    auto add = std::make_shared<ngraph::opset8::Add>(input_2, eps_const_pattern);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});
    auto divide = std::make_shared<ngraph::opset8::Divide>(input_1, max_or_add);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto eps_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(eps_const_pattern).get_node_shared_ptr());

        if (!eps_const) {
            return false;
        }
        for (float val : eps_const->get_vector<float>()) {
            if (val >= normalized_fp16_min) {
                return false;
            }
        }

        auto new_constant = std::make_shared<opset8::Constant>(eps_const->get_element_type(),
                                                               eps_const->get_shape(),
                                                               normalized_fp16_min);
        ngraph::copy_runtime_info(eps_const, new_constant);
        ngraph::replace_node(eps_const, new_constant);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide, matcher_name);
    register_matcher(m, callback);
}
