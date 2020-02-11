// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/add.hpp"

#include "ngraph_ops/group_conv_bias.hpp"
#include "ngraph/op/fused/group_conv.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

using namespace std;

namespace ngraph {
namespace pass {

class BatchNormDecomposition;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::BatchNormDecomposition: public ngraph::pass::GraphRewrite {
public:
    BatchNormDecomposition() : GraphRewrite() {
        batch_norm_decomposition();
    }

private:
    void batch_norm_decomposition();
};

void ngraph::pass::BatchNormDecomposition::batch_norm_decomposition() {
    Shape shape{2, 2, 1, 1};
    auto input = make_shared<pattern::op::Label>(element::f32, shape);
    auto mean_shape = Shape{2};
    auto mean = make_shared<pattern::op::Label>(element::f32, mean_shape);
    auto var_shape = Shape{2};
    auto var = make_shared<pattern::op::Label>(element::f32, var_shape);
    auto gamma_shape = Shape{2};
    auto gamma = make_shared<pattern::op::Label>(element::f32, gamma_shape);
    auto beta_shape = Shape{2};
    auto beta = make_shared<pattern::op::Label>(element::f32, beta_shape);
    auto bn = make_shared<op::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

    ngraph::graph_rewrite_callback callback = [input, gamma, beta, mean, var](ngraph::pattern::Matcher &m) {
        auto pattern_map = m.get_pattern_map();

        auto m_input = pattern_map[input];
        auto m_gamma = pattern_map[gamma];
        auto m_beta = pattern_map[beta];
        auto m_mean = pattern_map[mean];
        auto m_var = pattern_map[var];

        // TODO: check that all input shapes are static

        auto m_bn = dynamic_pointer_cast<op::BatchNormInference>(m.get_match_root());
        if (!m_bn) {
            return false;
        }

        // The code above represents this formulas
        //  scale = 1. / np.sqrt(variance + eps)
        //  shift = (mean * (-1)) * scale
        auto input_type = m_input->get_element_type();
        auto scale = make_shared<ngraph::op::v1::Divide>(
            op::Constant::create(input_type, Shape{}, {1}),
            make_shared<op::v1::Power>(
                make_shared<op::v1::Add>(
                    m_var,
                    op::Constant::create(input_type, Shape{}, {m_bn->get_eps_value()})),
                op::Constant::create(input_type, Shape{}, {0.5})));

        auto shift = make_shared<op::v1::Multiply>(
            scale,
            make_shared<op::v1::Multiply>(
                m_mean,
                op::Constant::create(m_input->get_element_type(), Shape{}, {-1})));

        // Expand Scale, Shift, Gamma and Beta to be aligned with layout
        size_t dims_to_add = m_input->get_shape().size() - 2;
        Shape gamma_shape = m_gamma->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i) gamma_shape.push_back(1);
        auto gamma_aligned = make_shared<op::v1::Reshape>(m_gamma, op::Constant::create(element::i64, Shape{gamma_shape.size()}, gamma_shape), true);

        Shape beta_shape = m_beta->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i) beta_shape.push_back(1);
        auto beta_aligned = make_shared<op::v1::Reshape>(m_beta, op::Constant::create(element::i64, Shape{beta_shape.size()}, beta_shape), true);

        Shape scale_shape = scale->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i) scale_shape.push_back(1);
        auto scale_aligned = make_shared<op::v1::Reshape>(scale, op::Constant::create(element::i64, Shape{scale_shape.size()}, scale_shape), true);

        Shape shift_shape = scale->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i) shift_shape.push_back(1);
        auto shift_aligned = make_shared<op::v1::Reshape>(shift, op::Constant::create(element::i64, Shape{shift_shape.size()}, shift_shape), true);

        // Connect: Mul(input, scale)->Add(mul, shift)->Mul(add, gamma)->Add(mul, beta)
        auto result = make_shared<op::v1::Add>(
            make_shared<op::v1::Multiply>(
                make_shared<op::v1::Add>(
                    make_shared<op::v1::Multiply>(
                        m_input,
                        scale_aligned),
                    shift_aligned),
                gamma_aligned),
            beta_aligned);

        result->set_friendly_name(m_bn->get_friendly_name());
        replace_node(m_bn, result);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, "BatchNormDecomposition");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
