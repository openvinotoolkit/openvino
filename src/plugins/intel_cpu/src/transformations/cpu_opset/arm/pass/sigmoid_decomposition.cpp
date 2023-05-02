// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "sigmoid_decomposition.hpp"

#include <openvino/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::SigmoidDecomposition::SigmoidDecomposition() {
    auto sigmoid = ngraph::pattern::wrap_type<opset4::Sigmoid>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto sigmoid = std::dynamic_pointer_cast<opset4::Sigmoid>(m.get_match_root());
        if (!sigmoid) {
            return false;
        }

        auto input = sigmoid->input_value(0);
        auto input_type = input.get_element_type();
        auto negative_one_constant = ov::opset4::Constant::create(input_type, ngraph::Shape{}, {-1.0});
        auto one_constant = ov::opset4::Constant::create(input_type, ngraph::Shape{}, {1.0});
        auto mul = std::make_shared<opset4::Multiply>(input, negative_one_constant);
        auto exp = std::make_shared<opset4::Exp>(mul);
        auto add = std::make_shared<opset4::Add>(exp, one_constant);
        auto divide = std::make_shared<opset4::Divide>(one_constant, add);

        mul->set_friendly_name(sigmoid->get_friendly_name());
        ngraph::copy_runtime_info(sigmoid, {mul, exp, add, divide});
        ngraph::replace_node(sigmoid, divide);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sigmoid, "SigmoidDecomposition");

    register_matcher(m, callback);
}
