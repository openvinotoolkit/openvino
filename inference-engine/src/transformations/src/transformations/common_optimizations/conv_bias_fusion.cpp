// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/conv_bias_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>
#include <functional>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>

#include <transformations/utils/utils.hpp>
#include "itt.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvFusion, "ConvFusion", 0);

template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ngraph::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

template <class Conv>
bool IsConvInLowPrecision(const std::shared_ptr<Conv>& conv) {
    if (!ngraph::is_type<ngraph::op::ConvolutionIE>(conv)) {
        return false;
    }

    auto isLowPrecision = [](const std::shared_ptr<ngraph::Node>& node, const size_t index) {
        const ngraph::element::Type inputType = node->get_input_element_type(index);
        return (inputType == ngraph::element::i8) || (inputType == ngraph::element::u8);
    };

    // Convolution operation has to be executed in INT8 if ...
    if (isLowPrecision(conv, 0) && isLowPrecision(conv, 1)) {
        // ... INT8 on activations && INT8 on weights
        return true;
    }

    const std::shared_ptr<ngraph::opset1::Subtract> subtract = ngraph::as_type_ptr<ngraph::opset1::Subtract>(conv->get_input_node_shared_ptr(0));
    if (subtract == nullptr) {
        return false;
    }

    // ... INT8 on activations with asymmetric quantization && INT8 on weights
    return isLowPrecision(subtract, 0) && isLowPrecision(subtract, 1) && isLowPrecision(conv, 1);
}

template <class Conv>
bool conv_callback(ngraph::pattern::Matcher &m) {
    auto eltwise = m.get_match_root();

    std::shared_ptr<ngraph::opset1::Constant> m_const;
    std::shared_ptr<Conv> m_conv;
    // FIXME: use auto [m_conv, m_const] when C++17 is available
    std::tie(m_conv, m_const) = parse_eltwise_inputs<Conv, ngraph::opset1::Constant>(eltwise);
    if (!m_conv || !m_const) {
        return false;
    }

    const auto & const_shape = m_const->get_shape();
    const auto & output_pshape = m_conv->get_output_partial_shape(0);

    if (output_pshape.rank().is_dynamic() || output_pshape[1].is_dynamic()) {
        return false;
    }

    const auto &  output_rank = output_pshape.rank().get_length();

    const int64_t channel_dim = output_pshape[1].get_length();

    bool is_scalar_multiplier(shape_size(const_shape) == 1);

    // Check that constant has shape [1, C, 1, 1] where the number of 1 is equal to
    // the number of spatial dimensions or it's a scalar. That means that Constant
    // applied per channel and can be fused into Convolution weights.
    // Also Constant shape rank must be less or equal Convolution output shape
    // otherwise fusion will break output broadcasting
    auto expected_shape = Shape(output_rank, 1);
    expected_shape[1] = channel_dim;

    if (op::util::check_for_broadcast(expected_shape, const_shape)) {
        return false;
    }

    // Broadcast constant to [1, C, 1, 1] where the number of 1 is equal to
    // the number of weights dimensions.
    Output<Node> final_const = m_const;
    if (is_scalar_multiplier) {
        final_const = op::util::broadcastTo(m_const, expected_shape);
    }

    if (final_const.get_shape().size() > 1) {
        final_const = std::make_shared<ngraph::opset1::Reshape>(final_const,
                                                                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {channel_dim}), true);
    }

    ngraph::Output<ngraph::Node> new_conv, new_weights, new_bias;
    if (std::dynamic_pointer_cast<ngraph::opset1::Add>(eltwise)) {
        // Fuse: ConvolutionIE/DeconvolutionIE->Add
        if (m_conv->inputs().size() == 2) {
            new_bias = final_const;
        } else {
            new_bias = std::make_shared<ngraph::opset1::Add>(final_const, m_conv->input_value(2));
        }
        new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), m_conv->input_value(1), new_bias});
    } else if (std::is_same<Conv, ngraph::op::ConvolutionIE>() && std::dynamic_pointer_cast<ngraph::opset1::Multiply>(eltwise) &&
               !IsConvInLowPrecision(m_conv)) {
        // Fuse: ConvolutionIE->Mul
        auto weights_shape = m_conv->input(1).get_shape();

        ngraph::Shape weights_const_shape(weights_shape.size(), 1);
        weights_const_shape[0] = weights_shape[0];

        auto const_reshape = std::make_shared<ngraph::opset1::Reshape>(final_const,
                                                                       ngraph::opset1::Constant::create(ngraph::element::i64,
                                                                                                        ngraph::Shape{weights_const_shape.size()},
                                                                                                        weights_const_shape),
                                                                       true);
        new_weights = std::make_shared<ngraph::opset1::Multiply> (m_conv->input_value(1), const_reshape);
        if (m_conv->inputs().size() == 2) {
            new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), new_weights});
        } else {
            auto bias_reshape = std::make_shared<ngraph::opset1::Reshape>(final_const,
                                                                          ngraph::opset1::Constant::create(ngraph::element::i64,
                                                                                                           ngraph::Shape{1},
                                                                                                           {weights_shape[0]}),
                                                                          true);
            new_bias = std::make_shared<ngraph::opset1::Multiply>(bias_reshape, final_const);
            new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), new_weights, new_bias});
        }
    } else {
        return false;
    }

    ngraph::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());
    new_conv.get_node_shared_ptr()->set_friendly_name(m.get_match_root()->get_friendly_name());
    ngraph::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvAddFusion, "ConvAddFusion", 0);

ngraph::pass::ConvAddFusion::ConvAddFusion() {
    MATCHER_SCOPE(ConvAddFusion);
    auto conv = ngraph::pattern::wrap_type<op::ConvolutionIE>(pattern::consumers_count(1));
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv, pattern::any_input()});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return conv_callback<op::ConvolutionIE>(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvMultiplyFusion, "ConvMultiplyFusion", 0);

ngraph::pass::ConvMultiplyFusion::ConvMultiplyFusion() {
    MATCHER_SCOPE(ConvMultiplyFusion);
    auto conv = ngraph::pattern::wrap_type<op::ConvolutionIE>(pattern::consumers_count(1));
    auto add = ngraph::pattern::wrap_type<opset1::Multiply>({conv, pattern::any_input()});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return conv_callback<op::ConvolutionIE>(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::DeconvAddFusion, "DeconvAddFusion", 0);

ngraph::pass::DeconvAddFusion::DeconvAddFusion() {
    MATCHER_SCOPE(DeconvAddFusion);
    auto conv = ngraph::pattern::wrap_type<op::DeconvolutionIE>(pattern::consumers_count(1));
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv, pattern::any_input()});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m){
        return conv_callback<op::DeconvolutionIE>(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}
