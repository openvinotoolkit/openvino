// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/conv_mul_fusion.hpp"

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset4.hpp>


bool check_shapes(const ngraph::Shape & ref_shape, const ngraph::Shape & other_shape) {
    // Check that other_shape doesn't broadcast ref_shape
    if (other_shape.size() > ref_shape.size()) {
        return false;
    }
    auto ref_it = ref_shape.rbegin();
    auto other_it = other_shape.rbegin();
    // Check that other_shape dims are equal to ref_shape dims
    // In case if other_shape rank is less than ref_shape rank
    // we stop comparision and return true
    while (other_it != other_shape.rend()) {
        if (*other_it != *ref_it && *other_it != 1) {
            return false;
        }
        ++other_it;
        ++ref_it;
    }
    return true;
}

ngraph::pass::ConvolutionMultiplyFusion::ConvolutionMultiplyFusion() {
    auto input = pattern::any_input();
    auto weights = ngraph::pattern::any_input(pattern::has_static_dim(0) /* has OIYX layout */);
    auto conv = ngraph::pattern::wrap_type<opset4::Convolution>({input, weights}, pattern::consumers_count(1));
    auto mul_const = ngraph::pattern::wrap_type<opset4::Constant>(pattern::has_static_shape());
    auto mul = ngraph::pattern::wrap_type<opset4::Multiply>({conv, mul_const});

    matcher_pass_callback callback = [conv, input, weights, mul, mul_const](pattern::Matcher & m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();

        const auto & m_weights = pattern_to_output.at(weights);
        const auto & m_const = pattern_to_output.at(mul_const);
        const auto & m_input = pattern_to_output.at(input);
        const auto & m_conv = pattern_to_output.at(conv).get_node_shared_ptr();
        const auto & m_mul = pattern_to_output.at(mul).get_node_shared_ptr();

        const auto & channel_dim = m_weights.get_partial_shape()[0].get_length();
        const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
        const auto & const_shape = m_const.get_shape();

        bool is_scalar_multiplier(shape_size(const_shape) == 1);

        // Check that constant has shape [1, C, 1, 1] where the number of 1 is equal to
        // the number of spatial dimensions or it's a scalar. That means that Constant
        // applied per channel and can be fused into Convolution weights.
        // Also Constant shape rank must be less or equal Convolution output shape
        // otherwise fusion will break output broadcasting
        auto expected_shape = Shape(weights_rank, 1);
        expected_shape[1] = channel_dim;

        if (!check_shapes(expected_shape, const_shape)) {
            return false;
        }

        // Reshape constant to [C, 1, 1, 1] where the number of 1 is equal to
        // the number of weights dimensions. In case of scalar we skip Reshape.
        // This Reshape aligns Constant shape for multiplication with weights.
        Output<Node> final_const = m_const;
        if (!is_scalar_multiplier) {
            auto final_const_shape = Shape(weights_rank, 1);
            final_const_shape[0] = channel_dim;
            final_const = std::make_shared<opset4::Reshape>(m_const,
                                                            opset4::Constant::create(ngraph::element::i64, ngraph::Shape{final_const_shape.size()},
                                                                                     final_const_shape), true);
        }

        // Multiply convolution weights with aligned Constant values
        auto weights_multiply = std::make_shared<opset4::Multiply>(m_weights, final_const);

        // Replace Convolution->Multiply with Convolution with new inputs
        auto new_conv = m_conv->copy_with_new_inputs({m_input, weights_multiply});
        new_conv->set_friendly_name(m_mul->get_friendly_name());
        copy_runtime_info({m_conv, m_mul}, {new_conv, final_const.get_node_shared_ptr(), weights_multiply});
        replace_node(m_mul, new_conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "ConvolutionMultiplyFusion");
    register_matcher(m, callback);
}

ngraph::pass::GroupConvolutionMultiplyFusion::GroupConvolutionMultiplyFusion() {
    auto input = pattern::any_input();
    auto weights = ngraph::pattern::any_input();//pattern::has_static_dims({0, 1}) /* has GOIYX layout */);
    auto conv = ngraph::pattern::wrap_type<opset4::GroupConvolution>({input, weights}, pattern::consumers_count(1));
    auto mul_const = ngraph::pattern::wrap_type<opset4::Constant>();//pattern::has_static_shape());
    auto mul = ngraph::pattern::wrap_type<opset4::Multiply>({conv, mul_const});

    matcher_pass_callback callback = [conv, input, weights, mul, mul_const](pattern::Matcher & m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();

        const auto & m_weights = pattern_to_output.at(weights);
        const auto & m_const = pattern_to_output.at(mul_const);
        const auto & m_input = pattern_to_output.at(input);
        const auto & m_conv = pattern_to_output.at(conv).get_node_shared_ptr();
        const auto & m_mul = pattern_to_output.at(mul).get_node_shared_ptr();

        const auto & G = m_weights.get_partial_shape()[0].get_length();
        const auto & O = m_weights.get_partial_shape()[1].get_length();
        const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
        const auto & const_shape = m_const.get_shape();

        bool is_scalar_multiplier(shape_size(const_shape) == 1);

        // Check that constant has shape [1, C (G * O), 1, 1] where the number of 1 is equal to
        // the number of spatial dimensions. That means that Constant applied per
        // channel and can be fused into Convolution weights.
        // Also Constant shape rank must be less or equal Convolution output shape
        // otherwise fusion will break output broadcasting
        auto expected_shape = Shape(weights_rank - 1, 1);
        expected_shape[1] = G * O;

        if (!check_shapes(expected_shape, const_shape)) {
            return false;
        }

        // Reshape constant to [G, O, 1, 1, 1] where the number of 1 is equal to
        // the number of weights dimensions. In case of scalar we skip Reshape.
        // This Reshape aligns Constant shape for multiplication with weights.
        Output<Node> final_const = m_const;
        if (!is_scalar_multiplier) {
            auto final_const_shape = Shape(weights_rank, 1);
            final_const_shape[0] = G;
            final_const_shape[1] = O;
            final_const = std::make_shared<opset4::Reshape>(m_const,
                                                            opset4::Constant::create(ngraph::element::i64, ngraph::Shape{final_const_shape.size()},
                                                                                     final_const_shape), true);
        }

        // Multiply convolution weights with aligned Constant values
        auto weights_multiply = std::make_shared<opset4::Multiply>(m_weights, final_const);

        // Replace Convolution->Multiply with Convolution with new inputs
        auto new_conv = m_conv->copy_with_new_inputs({m_input, weights_multiply});
        new_conv->set_friendly_name(m_mul->get_friendly_name());
        copy_runtime_info({m_conv, m_mul}, {new_conv, final_const.get_node_shared_ptr(), weights_multiply});
        replace_node(m_mul, new_conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "GroupConvolutionMultiplyFusion");
    register_matcher(m, callback);
}

ngraph::pass::ConvolutionBackpropDataMultiplyFusion::ConvolutionBackpropDataMultiplyFusion() {
    auto input = pattern::any_input();
    auto weights = ngraph::pattern::any_input(pattern::has_static_dim(1) /* has IOYX layout */);
    auto conv = ngraph::pattern::wrap_type<opset4::ConvolutionBackpropData>({input, weights}, pattern::consumers_count(1));
    auto mul_const = ngraph::pattern::wrap_type<opset4::Constant>(pattern::has_static_shape());
    auto mul = ngraph::pattern::wrap_type<opset4::Multiply>({conv, mul_const});

    matcher_pass_callback callback = [conv, input, weights, mul, mul_const](pattern::Matcher & m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();

        const auto & m_weights = pattern_to_output.at(weights);
        const auto & m_const = pattern_to_output.at(mul_const);
        const auto & m_input = pattern_to_output.at(input);
        const auto & m_conv = pattern_to_output.at(conv).get_node_shared_ptr();
        const auto & m_mul = pattern_to_output.at(mul).get_node_shared_ptr();

        const auto & channel_dim = m_weights.get_partial_shape()[1].get_length();
        const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
        const auto & const_shape = m_const.get_shape();

        bool is_scalar_multiplier(shape_size(const_shape) == 1);

        // Check that constant has shape [1, C, 1, 1] where the number of 1 is equal to
        // the number of spatial dimensions. That means that Constant applied per
        // channel and can be fused into Convolution weights.
        // Also Constant shape rank must be less or equal Convolution output shape
        // otherwise fusion will break output broadcasting
        auto expected_shape = Shape(weights_rank, 1);
        expected_shape[1] = channel_dim;

        if (!check_shapes(expected_shape, const_shape)) {
            return false;
        }

        // Reshape constant to [O, 1, 1] where the number of 1 is equal to
        // the number of weights dimensions minus 1 (input dimension).
        // This Reshape aligns Constant shape for multiplication with weights.
        Output<Node> final_const = m_const;
        if (!is_scalar_multiplier) {
            auto final_const_shape = Shape(weights_rank - 1, 1);
            final_const_shape[0] = channel_dim;
            final_const = std::make_shared<opset4::Reshape>(m_const,
                                                            opset4::Constant::create(ngraph::element::i64, ngraph::Shape{final_const_shape.size()},
                                                                                     final_const_shape), true);
        }

        // Multiply convolution weights with aligned Constant values
        auto weights_multiply = std::make_shared<opset4::Multiply>(m_weights, final_const);

        // Replace Convolution->Multiply with Convolution with new inputs
        auto new_conv = m_conv->copy_with_new_inputs({m_input, weights_multiply});
        new_conv->set_friendly_name(m_mul->get_friendly_name());
        copy_runtime_info({m_conv, m_mul}, {new_conv, final_const.get_node_shared_ptr(), weights_multiply});
        replace_node(m_mul, new_conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "ConvolutionBackpropDataMultiplyFusion");
    register_matcher(m, callback);
}

ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion::GroupConvolutionBackpropDataMultiplyFusion() {
    auto input = pattern::any_input();
    auto weights = ngraph::pattern::any_input(pattern::has_static_dims({0, 2}) /* has GIOYX layout */);
    auto conv = ngraph::pattern::wrap_type<opset4::GroupConvolutionBackpropData>({input, weights}, pattern::consumers_count(1));
    auto mul_const = ngraph::pattern::wrap_type<opset4::Constant>(pattern::has_static_shape());
    auto mul = ngraph::pattern::wrap_type<opset4::Multiply>({conv, mul_const});

    matcher_pass_callback callback = [conv, input, weights, mul, mul_const](pattern::Matcher & m) -> bool {
        const auto & pattern_to_output = m.get_pattern_value_map();

        const auto & m_weights = pattern_to_output.at(weights);
        const auto & m_const = pattern_to_output.at(mul_const);
        const auto & m_input = pattern_to_output.at(input);
        const auto & m_conv = pattern_to_output.at(conv).get_node_shared_ptr();
        const auto & m_mul = pattern_to_output.at(mul).get_node_shared_ptr();

        const auto & G = m_weights.get_partial_shape()[0].get_length();
        const auto & O = m_weights.get_partial_shape()[2].get_length();
        const auto & weights_rank = m_weights.get_partial_shape().rank().get_length();
        const auto & const_shape = m_const.get_shape();

        bool is_scalar_multiplier(shape_size(const_shape) == 1);

        // Check that constant has shape [1, C (G * O), 1, 1] where the number of 1 is equal to
        // the number of spatial dimensions. That means that Constant applied per
        // channel and can be fused into Convolution weights.
        // Also Constant shape rank must be less or equal Convolution output shape
        // otherwise fusion will break output broadcasting
        auto expected_shape = Shape(weights_rank - 1, 1);
        expected_shape[1] = G * O;

        if (!check_shapes(expected_shape, const_shape)) {
            return false;
        }

        // Reshape constant to [G, 1, O, 1, 1, 1] where the number of 1 is equal to
        // the number of weights dimensions. In case of scalar we skip Reshape.
        // This Reshape aligns Constant shape for multiplication with weights.
        Output<Node> final_const = m_const;
        if (!is_scalar_multiplier) {
            auto final_const_shape = Shape(weights_rank, 1);
            final_const_shape[0] = G;
            final_const_shape[2] = O;
            final_const = std::make_shared<opset4::Reshape>(m_const,
                                                            opset4::Constant::create(ngraph::element::i64, ngraph::Shape{final_const_shape.size()},
                                                                                     final_const_shape), true);
        }

        // Multiply convolution weights with aligned Constant values
        auto weights_multiply = std::make_shared<opset4::Multiply>(m_weights, final_const);

        // Replace Convolution->Multiply with Convolution with new inputs
        auto new_conv = m_conv->copy_with_new_inputs({m_input, weights_multiply});
        new_conv->set_friendly_name(m_mul->get_friendly_name());
        copy_runtime_info({m_conv, m_mul}, {new_conv, final_const.get_node_shared_ptr(), weights_multiply});
        replace_node(m_mul, new_conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "GroupConvolutionMultiplyFusion");
    register_matcher(m, callback);
}