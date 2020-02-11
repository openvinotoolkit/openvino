// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>
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

#include <ngraph/pass/graph_rewrite.hpp>

#include "mul_add_squence_fusion.hpp"

namespace ngraph {
namespace pass {

class ConvFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvFusion: public ngraph::pass::GraphRewrite {
public:
    ConvFusion() : GraphRewrite() {
        fuse_convolution_with<op::ConvolutionIE,   op::v1::Multiply>();
        fuse_convolution_with<op::ConvolutionIE,   op::v1::Add>();
        fuse_convolution_with<op::DeconvolutionIE, op::v1::Add>();
    }

private:
    template <class Conv, class Eltwise>
    void fuse_convolution_with();

    template <class Conv>
    ngraph::graph_rewrite_callback get_callback();
};

template <class Conv, class Eltwise>
void ngraph::pass::ConvFusion::fuse_convolution_with() {
    static_assert(std::is_same<Eltwise, ngraph::op::v1::Multiply>() || std::is_same<Eltwise, ngraph::op::v1::Add>(),
                  "This transformation works only with ngraph::op::v1::Add and ngraph::op::v1::Multiply");

    static_assert(std::is_same<Conv, ngraph::op::ConvolutionIE>() || std::is_same<Conv, ngraph::op::DeconvolutionIE>(),
                  "This transformation works only with ngraph::op::ConvolutionIE and ngraph::op::DeconvolutionIE");

    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto filters = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{1});

    auto conv = std::make_shared<Conv>(data_batch,
                                       filters,
                                       Strides{1, 1},
                                       CoordinateDiff{0, 0},
                                       CoordinateDiff{0, 0},
                                       Strides{1, 1},
                                       Shape{1, 1, 1, 1});
    auto last = std::make_shared<Eltwise>(conv, bias);

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
        auto constant_shape = m_const->get_shape();
        auto output_shape = m_conv->get_shape();
        size_t constant_size = std::accumulate(constant_shape.begin(), constant_shape.end(), 1, std::multiplies<size_t>());
        if (constant_size != output_shape[1]) {
            return false;
        }

        std::shared_ptr<ngraph::Node> constant(m_const);

        if (constant_shape.size() > 1) {
            constant = std::make_shared<op::v1::Reshape>(constant, op::Constant::create(element::i64, Shape{1}, {output_shape[1]}), true);
        }

        if (m_conv->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        std::shared_ptr<Node> new_conv, new_weights, new_bias;
        if (std::dynamic_pointer_cast<op::v1::Add>(eltwise)) {
            // Fuse: ConvolutionIE/DeconvolutionIE->Add
            if (m_conv->inputs().size() == 2) {
                new_bias = constant;
            } else {
                new_bias = std::make_shared<op::v1::Add>(constant, m_conv->input_value(2));
            }
            new_conv = m_conv->copy({m_conv->input_value(0), m_conv->input_value(1), new_bias});
        } else if (std::is_same<Conv, op::ConvolutionIE>() && std::dynamic_pointer_cast<op::v1::Multiply>(eltwise)) {
            // Fuse: ConvolutionIE->Mul
            auto weights_shape = m_conv->input(1).get_shape();

            Shape const_shape(weights_shape.size(), 1);
            const_shape[0] = weights_shape[0];

            auto const_reshape = std::make_shared<op::v1::Reshape>(constant,
                                                                   op::Constant::create(element::i64, Shape{const_shape.size()}, const_shape), true);
            new_weights = std::make_shared<op::v1::Multiply> (m_conv->input_value(1), const_reshape);
            if (m_conv->inputs().size() == 2) {
                new_conv = m_conv->copy({m_conv->input_value(0), new_weights});
            } else {
                auto bias_reshape = std::make_shared<op::v1::Reshape>(constant, op::Constant::create(element::i64, Shape{1}, {weights_shape[0]}), true);
                new_bias = std::make_shared<op::v1::Multiply>(bias_reshape, constant);
                new_conv = m_conv->copy({m_conv->input_value(0), new_weights, new_bias});
            }
        } else {
            return false;
        }

        new_conv->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), new_conv);
        return true;
    };
    return callback;
}

