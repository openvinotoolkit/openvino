// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/add.hpp"

#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/rt_info.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

#include "transformations/mul_add_squence_fusion.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvFusion: public ngraph::pass::GraphRewrite {
public:
    ConvFusion() : GraphRewrite() {
        fuse_convolution_with<op::ConvolutionIE,   opset1::Multiply>();
        fuse_convolution_with<op::ConvolutionIE,   opset1::Add>();
        fuse_convolution_with<op::DeconvolutionIE, opset1::Add>();
    }

private:
    template <class Conv, class Eltwise>
    void fuse_convolution_with();

    template <class Conv>
    ngraph::graph_rewrite_callback get_callback();
};

template <class Conv, class Eltwise>
void ngraph::pass::ConvFusion::fuse_convolution_with() {
    static_assert(std::is_same<Eltwise, ngraph::opset1::Multiply>() || std::is_same<Eltwise, ngraph::opset1::Add>(),
                  "This transformation works only with ngraph::opset1::Add and ngraph::opset1::Multiply");

    static_assert(std::is_same<Conv, ngraph::op::ConvolutionIE>() || std::is_same<Conv, ngraph::op::DeconvolutionIE>(),
                  "This transformation works only with ngraph::op::ConvolutionIE and ngraph::op::DeconvolutionIE");

    auto conv = std::make_shared<pattern::op::Label>(element::f32, Shape{},
            [](const std::shared_ptr<Node> & node) -> bool {
                return std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node) ||
                       std::dynamic_pointer_cast<ngraph::op::DeconvolutionIE>(node);
    });

    auto last = std::make_shared<Eltwise>(conv, std::make_shared<pattern::op::Label>(element::f32, Shape{1}));

    auto m = std::make_shared<ngraph::pattern::Matcher>(last, "ConvFusion");
    this->add_matcher(m, get_callback<Conv>(), PassProperty::CHANGE_DYNAMIC_STATE);
}

template <class Conv>
ngraph::graph_rewrite_callback ngraph::pass::ConvFusion::get_callback() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        auto eltwise = m.get_match_root();

        std::shared_ptr<op::Constant> m_const;
        std::shared_ptr<Conv> m_conv;
        // FIXME: use auto [m_conv, m_const] when C++17 is available
        std::tie(m_conv, m_const) = parse_eltwise_inputs<Conv, op::Constant>(eltwise);
        if (!m_conv || !m_const) {
            return false;
        }

        // TODO: check that constant can be scalar and do not match [1, C, 1, 1] layout
        const auto constant_shape = m_const->get_shape();
        const auto output_pshape = m_conv->get_output_partial_shape(0);

        if (output_pshape.rank().is_dynamic() || output_pshape[1].is_dynamic()) {
            return false;
        }

        const auto channel_dim = output_pshape[1].get_length();

        size_t constant_size = std::accumulate(constant_shape.begin(), constant_shape.end(), 1, std::multiplies<size_t>());
        if (constant_size != channel_dim) {
            return false;
        }

        Output<Node> constant(m_const);

        if (constant_shape.size() > 1) {
            constant = std::make_shared<opset1::Reshape>(constant, op::Constant::create(element::i64, Shape{1}, {channel_dim}), true);
        }

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        Output<Node> new_conv, new_weights, new_bias;
        if (std::dynamic_pointer_cast<opset1::Add>(eltwise)) {
            // Fuse: ConvolutionIE/DeconvolutionIE->Add
            if (m_conv->inputs().size() == 2) {
                new_bias = constant;
            } else {
                new_bias = std::make_shared<opset1::Add>(constant, m_conv->input_value(2));
            }
            new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), m_conv->input_value(1), new_bias});
        } else if (std::is_same<Conv, op::ConvolutionIE>() && std::dynamic_pointer_cast<opset1::Multiply>(eltwise) && false) {
            // Fuse: ConvolutionIE->Mul
            auto weights_shape = m_conv->input(1).get_shape();

            Shape const_shape(weights_shape.size(), 1);
            const_shape[0] = weights_shape[0];

            auto const_reshape = std::make_shared<opset1::Reshape>(constant,
                                                                   op::Constant::create(element::i64, Shape{const_shape.size()}, const_shape), true);
            new_weights = std::make_shared<opset1::Multiply> (m_conv->input_value(1), const_reshape);
            if (m_conv->inputs().size() == 2) {
                new_conv = m_conv->clone_with_new_inputs({m_conv->input_value(0), new_weights});
            } else {
                auto bias_reshape = std::make_shared<opset1::Reshape>(constant, op::Constant::create(element::i64, Shape{1}, {weights_shape[0]}), true);
                new_bias = std::make_shared<opset1::Multiply>(bias_reshape, constant);
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

