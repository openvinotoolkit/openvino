// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset8.hpp>

#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyConvolutionFusion, "MultiplyConvolutionFusion", 0);

static bool is_dequantization_subgraph(const Output<Node>& multiply) {
    auto inputs = multiply.get_node()->input_values();
    const auto subtract = std::find_if(inputs.begin(), inputs.end(),
                                       [] (const Output<Node>& n) -> bool {
                                           return ov::is_type<ngraph::opset8::Subtract>(n.get_node());
                                       });
    if (subtract != inputs.end())
        inputs = subtract->get_node()->input_values();
    const auto first_convert = std::find_if(inputs.begin(), inputs.end(),
                                      [] (const Output<Node>& n) -> bool {
                                          if (ov::is_type<ngraph::opset8::Convert>(n.get_node())) {
                                              const auto input = n.get_node()->input_value(0);
                                              return ov::is_type<ngraph::opset8::Convert>(input.get_node());
                                          }
                                          return false;
                                      });
    if (first_convert == inputs.end())
        return false;
    const auto second_convert = first_convert->get_node()->input_value(0);
    const element::Type& first_convert_src_type = second_convert.get_element_type();
    const element::Type& first_convert_dest_type = first_convert->get_element_type();
    const element::Type second_convert_src_type = second_convert.get_node()->input_value(0).get_element_type();
    return (first_convert_src_type == element::i8 || first_convert_src_type == element::u8) &&
        first_convert_dest_type == second_convert_src_type;
}

ngraph::pass::MultiplyConvolutionFusion::MultiplyConvolutionFusion() {
    MATCHER_SCOPE(MultiplyConvolutionFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({input_pattern, mul_const_pattern}, pattern::consumers_count(1));
    auto weights_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ngraph::pattern::wrap_type<opset8::Convolution>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        if (is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        const auto& mul_const = pattern_to_output.at(mul_const_pattern);

        const auto& weights_shape = weights.get_shape();
        const auto& mul_const_shape = mul_const.get_shape();
        if (op::util::check_for_broadcast(weights_shape, mul_const_shape) ||
            (weights_shape.size() == mul_const_shape.size() && mul_const_shape[0] != 1)) {
            return false;
        }

        auto weights_multiply = std::make_shared<opset8::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& input = pattern_to_output.at(input_pattern);
        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()},
                          {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyGroupConvolutionFusion, "MultiplyGroupConvolutionFusion", 0);

ngraph::pass::MultiplyGroupConvolutionFusion::MultiplyGroupConvolutionFusion() {
    MATCHER_SCOPE(MultiplyGroupConvolutionFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({input_pattern, mul_const_pattern}, pattern::consumers_count(1));
    auto weights_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ngraph::pattern::wrap_type<opset8::GroupConvolution>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        if (is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();

        const auto& weights_shape = weights.get_shape();
        auto mul_const_shape = mul_const->get_shape();
        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            if (weights_shape.size() - mul_const_shape.size() > 1)
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size() - 1, 1);
            if (mul_const_shape[0] != 1)
                return false;
            auto G = mul_const_shape[1] > 1 ? weights_shape[0] : 1;
            auto C = mul_const_shape[1] / G;
            Shape new_shape{G, 1, C};
            std::copy(mul_const_shape.begin() + 2, mul_const_shape.end(), std::back_inserter(new_shape));
            if (op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<opset8::Reshape>(mul_const, op::Constant::create(element::u64, Shape{new_shape.size()}, new_shape), false);
        }

        auto weights_multiply = std::make_shared<opset8::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& input = pattern_to_output.at(input_pattern);
        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()},
                          {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyConvolutionBackpropDataFusion, "MultiplyConvolutionBackpropDataFusion", 0);

ngraph::pass::MultiplyConvolutionBackpropDataFusion::MultiplyConvolutionBackpropDataFusion() {
    MATCHER_SCOPE(MultiplyConvolutionBackpropDataFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({input_pattern, mul_const_pattern}, pattern::consumers_count(1));
    auto weights_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ngraph::pattern::wrap_type<opset8::ConvolutionBackpropData>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        if (is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        const auto& weights_shape = weights.get_shape();
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();

        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            if (weights_shape.size() > mul_const_shape.size())
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size(), 1);
            for (size_t i = 0; i < mul_const_shape.size(); i++) {
                if (i == 1)
                   continue;
                if (mul_const_shape[i] != 1)
                    return false;
            }
            Shape new_shape{mul_const_shape[1], 1};
            std::copy(mul_const_shape.begin() + 2, mul_const_shape.end(), std::back_inserter(new_shape));
            if (op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<opset8::Reshape>(mul_const, op::Constant::create(element::u64, Shape{new_shape.size()}, new_shape), false);
        }

        auto weights_multiply = std::make_shared<opset8::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& input = pattern_to_output.at(input_pattern);
        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()},
                          {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion, "MultiplyGroupConvolutionBackpropDataFusion", 0);

ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion::MultiplyGroupConvolutionBackpropDataFusion() {
    MATCHER_SCOPE(MultiplyGroupConvolutionBackpropDataFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({input_pattern, mul_const_pattern}, pattern::consumers_count(1));
    auto weights_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ngraph::pattern::wrap_type<opset8::GroupConvolutionBackpropData>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        if (is_dequantization_subgraph(pattern_to_output.at(mul_pattern)))
            return false;

        const auto& weights = pattern_to_output.at(weights_pattern);
        std::shared_ptr<Node> mul_const = pattern_to_output.at(mul_const_pattern).get_node_shared_ptr();

        const auto& weights_shape = weights.get_shape();
        auto mul_const_shape = mul_const->get_shape();
        if (shape_size(mul_const->get_shape()) > 1) {
            auto mul_const_shape = mul_const->get_shape();
            if (weights_shape.size() - mul_const_shape.size() > 1)
                mul_const_shape.insert(mul_const_shape.begin(), weights_shape.size() - mul_const_shape.size() - 1, 1);
            for (size_t i = 0; i < mul_const_shape.size(); i++) {
                if (i == 1)
                   continue;
                if (mul_const_shape[i] != 1)
                    return false;
            }
            auto G = mul_const_shape[1] > 1 ? weights_shape[0] : 1;
            auto C = mul_const_shape[1] / G;
            Shape new_shape{G, C, 1};
            std::copy(mul_const_shape.begin() + 2, mul_const_shape.end(), std::back_inserter(new_shape));
            if (op::util::check_for_broadcast(weights_shape, new_shape)) {
                return false;
            }
            mul_const = std::make_shared<opset8::Reshape>(mul_const, op::Constant::create(element::u64, Shape{new_shape.size()}, new_shape), false);
        }

        auto weights_multiply = std::make_shared<opset8::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& input = pattern_to_output.at(input_pattern);
        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()},
                          {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}
