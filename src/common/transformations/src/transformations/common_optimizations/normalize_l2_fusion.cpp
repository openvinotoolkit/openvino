// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_l2_fusion.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::NormalizeL2Fusion::NormalizeL2Fusion() {
    MATCHER_SCOPE(NormalizeL2Fusion);
    auto input = ngraph::pattern::any_input();

    auto exp = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto pow = std::make_shared<ngraph::opset8::Power>(input, exp);
    auto axes = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto reduce_sum = std::make_shared<ngraph::opset8::ReduceSum>(pow, axes);
    auto eps_const = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();

    auto max = std::make_shared<ngraph::opset8::Maximum>(reduce_sum, eps_const);
    auto add = std::make_shared<ngraph::opset8::Add>(reduce_sum, eps_const);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});

    // Sqrt can be as Sqrt node or as Power node with exponent 2
    auto sqrt = std::make_shared<ngraph::opset8::Sqrt>(max_or_add);
    auto exp2 = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto pow_as_sqrt = std::make_shared<ngraph::opset8::Power>(max_or_add, exp2);
    auto power_or_sqrt = std::make_shared<pattern::op::Or>(OutputVector{sqrt, pow_as_sqrt});

    // divide(input,sqrt(..)) can be as mul(input, power(..., -0.5f))
    auto divide = std::make_shared<ngraph::opset8::Divide>(input, power_or_sqrt);
    auto exp3 = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto un_sqrt = std::make_shared<ngraph::opset8::Power>(max_or_add, exp3);
    auto mul = std::make_shared<ngraph::opset8::Multiply>(input, un_sqrt);
    auto divide_or_mul = std::make_shared<pattern::op::Or>(OutputVector{divide, mul});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        // its value is true if last node of patter will be Divide node
        bool is_divide_variant = pattern_to_output.count(divide);
        if (!is_divide_variant && !pattern_to_output.count(mul)) {
            return false;
        }
        // Its value is true if Sqrt willl be Sqrt Node(not Power(..., 0.5f))
        bool sqrt_as_sqrt = false;
        if (is_divide_variant) {
            sqrt_as_sqrt = pattern_to_output.count(sqrt);
        }

        const auto data_input = pattern_to_output.at(input);
        const auto exp_input =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(exp).get_node_shared_ptr());
        const auto axes_input =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());
        const auto eps_attr =
            std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());
        const auto exp2_input =
            (is_divide_variant && !sqrt_as_sqrt)
                ? std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(exp2).get_node_shared_ptr())
                : nullptr;
        const auto exp3_input =
            (!is_divide_variant)
                ? std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(exp3).get_node_shared_ptr())
                : nullptr;

        if (!exp_input || !axes_input || !eps_attr) {
            return false;
        }

        if (exp2_input && !op::util::has_constant_value<float>(exp2_input, 0.5f)) {
            return false;
        }

        if (exp3_input && !op::util::has_constant_value<float>(exp3_input, -0.5f)) {
            return false;
        }

        const bool is_square_pow = op::util::has_constant_value<float>(exp_input, 2.0f);
        if (!is_square_pow) {
            return false;
        }
        if (shape_size(eps_attr->get_shape()) > 1) {
            return false;
        }
        const auto eps_attr_value = eps_attr->cast_vector<float>()[0];

        op::EpsMode mode;
        Output<Node> eps_node;
        if (pattern_to_output.count(max)) {
            mode = op::EpsMode::MAX;
            eps_node = pattern_to_output.at(max);
        } else if (pattern_to_output.count(add)) {
            mode = op::EpsMode::ADD;
            eps_node = pattern_to_output.at(add);
        } else {
            return false;
        }

        auto normalize_l2 = std::make_shared<ngraph::opset8::NormalizeL2>(data_input, axes_input, eps_attr_value, mode);
        if (transformation_callback(normalize_l2))
            return false;

        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());
        if (is_divide_variant) {
            if (sqrt_as_sqrt) {
                ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                           pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                           pattern_to_output.at(sqrt).get_node_shared_ptr(),
                                           pattern_to_output.at(divide).get_node_shared_ptr(),
                                           eps_node.get_node_shared_ptr()},
                                          normalize_l2);
            } else {
                ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                           pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                           pattern_to_output.at(pow_as_sqrt).get_node_shared_ptr(),
                                           pattern_to_output.at(divide).get_node_shared_ptr(),
                                           eps_node.get_node_shared_ptr()},
                                          normalize_l2);
            }
        } else {
            ngraph::copy_runtime_info({pattern_to_output.at(pow).get_node_shared_ptr(),
                                       pattern_to_output.at(reduce_sum).get_node_shared_ptr(),
                                       pattern_to_output.at(un_sqrt).get_node_shared_ptr(),
                                       pattern_to_output.at(mul).get_node_shared_ptr(),
                                       eps_node.get_node_shared_ptr()},
                                      normalize_l2);
        }
        ngraph::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_or_mul, matcher_name);
    register_matcher(m, callback);
}
