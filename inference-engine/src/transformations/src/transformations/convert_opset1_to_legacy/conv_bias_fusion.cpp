// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>

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
ngraph::graph_rewrite_callback get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        auto eltwise = m.get_match_root();

        std::shared_ptr<ngraph::opset1::Constant> m_const;
        std::shared_ptr<Conv> m_conv;
        // FIXME: use auto [m_conv, m_const] when C++17 is available
        std::tie(m_conv, m_const) = parse_eltwise_inputs<Conv, ngraph::opset1::Constant>(eltwise);
        if (!m_conv || !m_const) {
            return false;
        }

        // TODO: check that constant can be scalar and do not match [1, C, 1, 1] layout
        const auto constant_shape = m_const->get_shape();
        const auto output_pshape = m_conv->get_output_partial_shape(0);

        if (output_pshape.rank().is_dynamic() || output_pshape[1].is_dynamic()) {
            return false;
        }

        const int64_t channel_dim = output_pshape[1].get_length();

        size_t constant_size = std::accumulate(constant_shape.begin(), constant_shape.end(), 1, std::multiplies<size_t>());
        if (constant_size != channel_dim) {
            return false;
        }

        ngraph::Output<ngraph::Node> constant(m_const);

        if (constant_shape.size() > 1) {
            constant = std::make_shared<ngraph::opset1::Reshape>(constant,
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {channel_dim}), true);
        }

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        ngraph::Output<ngraph::Node> new_conv, new_weights, new_bias;
        if (std::dynamic_pointer_cast<ngraph::opset1::Add>(eltwise)) {
            // Fuse: ConvolutionIE/DeconvolutionIE->Add
            if (m_conv->inputs().size() == 2) {
                new_bias = constant;
            } else {
                new_bias = std::make_shared<ngraph::opset1::Add>(constant, m_conv->input_value(2));
            }
            new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), m_conv->input_value(1), new_bias});
        } else if (std::is_same<Conv, ngraph::op::ConvolutionIE>() && std::dynamic_pointer_cast<ngraph::opset1::Multiply>(eltwise)) {
            // Fuse: ConvolutionIE->Mul
            auto weights_shape = m_conv->input(1).get_shape();

            ngraph::Shape const_shape(weights_shape.size(), 1);
            const_shape[0] = weights_shape[0];

            auto const_reshape = std::make_shared<ngraph::opset1::Reshape>(constant,
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{const_shape.size()}, const_shape), true);
            new_weights = std::make_shared<ngraph::opset1::Multiply> (m_conv->input_value(1), const_reshape);
            if (m_conv->inputs().size() == 2) {
                new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), new_weights});
            } else {
                auto bias_reshape = std::make_shared<ngraph::opset1::Reshape>(constant,
                        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {weights_shape[0]}), true);
                new_bias = std::make_shared<ngraph::opset1::Multiply>(bias_reshape, constant);
                new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), new_weights, new_bias});
            }
        } else {
            return false;
        }

        ngraph::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());
        new_conv.get_node_shared_ptr()->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
        return true;
    };
    return callback;
}

ngraph::pass::ConvAddFusion::ConvAddFusion() {
    auto conv = ngraph::pattern::wrap_type<op::ConvolutionIE>();
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv, std::make_shared<pattern::op::Label>()});

    matcher_pass_callback callback = get_callback<op::ConvolutionIE>();

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "ConvAddFusion");
    register_matcher(m, callback);
}

ngraph::pass::ConvMultiplyFusion::ConvMultiplyFusion() {
    auto conv = ngraph::pattern::wrap_type<op::ConvolutionIE>();
    auto add = ngraph::pattern::wrap_type<opset1::Multiply>({conv, std::make_shared<pattern::op::Label>()});

    matcher_pass_callback callback = get_callback<op::ConvolutionIE>();

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "ConvMultiplyFusion");
    register_matcher(m, callback);
}

ngraph::pass::DeconvAddFusion::DeconvAddFusion() {
    auto conv = ngraph::pattern::wrap_type<op::DeconvolutionIE>();
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv, std::make_shared<pattern::op::Label>()});

    matcher_pass_callback callback = get_callback<op::DeconvolutionIE>();

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "DeconvAddFusion");
    register_matcher(m, callback);
}
