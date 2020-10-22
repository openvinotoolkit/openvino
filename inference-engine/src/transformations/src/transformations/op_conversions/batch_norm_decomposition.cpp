// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/batch_norm_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::BatchNormDecomposition, "BatchNormDecomposition", 0);

ngraph::pass::BatchNormDecomposition::BatchNormDecomposition() {
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
    auto bn = make_shared<opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

    ngraph::graph_rewrite_callback callback = [this, input, gamma, beta, mean, var](ngraph::pattern::Matcher &m) {
        auto pattern_map = m.get_pattern_map();

        auto m_input = pattern_map[input];
        auto m_gamma = pattern_map[gamma];
        auto m_beta = pattern_map[beta];
        auto m_mean = pattern_map[mean];
        auto m_var = pattern_map[var];

        // TODO: check that all input shapes are static

        auto m_bn = dynamic_pointer_cast<opset1::BatchNormInference>(m.get_match_root());
        if (!m_bn) {
            return false;
        }

        const auto& input_type = m_input->get_element_type();
        // scale_add = variance + eps
        auto scale_add = make_shared<opset5::Add>(m_var, opset5::Constant::create(input_type, Shape{}, {m_bn->get_eps_value()}));
        // scale = sqrt(variance + eps)
        auto scale = make_shared<opset5::Sqrt>(scale_add);
        // Divide `gamma` by `sqrt(variance + eps)`
        auto gamma_div_scale = std::make_shared<opset5::Divide>(m_gamma, scale);

        size_t dims_to_add = m_input->get_shape().size() - 2;
        Shape input_aligned_shape = m_gamma->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i)
            input_aligned_shape.push_back(1);
        auto new_shape = opset5::Constant::create(element::i64, Shape{input_aligned_shape.size()}, input_aligned_shape);

        auto gamma_div_scale_aligned = make_shared<opset5::Reshape>(gamma_div_scale, new_shape, true);
        auto beta_aligned = make_shared<opset5::Reshape>(m_beta, new_shape, true);
        auto mean_aligned = make_shared<opset5::Reshape>(m_mean, new_shape, true);

        // input_sub_mean = input - mean
        auto input_sub_mean = register_new_node<opset5::Subtract>(m_input, mean_aligned);
        // Multiply  `input - mean` and `gamma / sqrt(variance + eps)`
        auto mul = std::make_shared<opset5::Multiply>(input_sub_mean, gamma_div_scale_aligned);
        // Add `(input - mean) * gamma / sqrt(variance + eps)` and `beta`
        auto add = std::make_shared<opset5::Add>(mul, beta_aligned);

        add->set_friendly_name(m_bn->get_friendly_name());

        copy_runtime_info(m_bn, {scale_add, scale, gamma_div_scale, gamma_div_scale_aligned,
            beta_aligned, input_sub_mean, mul, add});

        replace_node(m_bn, add);

        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, "BatchNormDecomposition");
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::BatchNormV5Decomposition, "BatchNormDecomposition", 5);

ngraph::pass::BatchNormV5Decomposition::BatchNormV5Decomposition() {
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
    auto bn = make_shared<opset5::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

    ngraph::graph_rewrite_callback callback = [this, input, gamma, beta, mean, var](ngraph::pattern::Matcher &m) {
        auto pattern_map = m.get_pattern_map();

        auto m_input = pattern_map[input];
        auto m_gamma = pattern_map[gamma];
        auto m_beta = pattern_map[beta];
        auto m_mean = pattern_map[mean];
        auto m_var = pattern_map[var];

        // TODO: check that all input shapes are static

        auto m_bn = dynamic_pointer_cast<opset5::BatchNormInference>(m.get_match_root());
        if (!m_bn) {
            return false;
        }

        const auto& input_type = m_input->get_element_type();
        // scale_add = variance + eps
        auto scale_add = make_shared<opset5::Add>(m_var, opset5::Constant::create(input_type, Shape{}, {m_bn->get_eps_value()}));
        // scale = sqrt(variance + eps)
        auto scale = make_shared<opset5::Sqrt>(scale_add);
        // Divide `gamma` by `sqrt(variance + eps)`
        auto gamma_div_scale = std::make_shared<opset5::Divide>(m_gamma, scale);

        size_t dims_to_add = m_input->get_shape().size() - 2;
        Shape input_aligned_shape = m_gamma->get_shape();
        for (size_t i = 0; i < dims_to_add; ++i)
            input_aligned_shape.push_back(1);
        auto new_shape = opset5::Constant::create(element::i64, Shape{input_aligned_shape.size()}, input_aligned_shape);

        auto gamma_div_scale_aligned = make_shared<opset5::Reshape>(gamma_div_scale, new_shape, true);
        auto beta_aligned = make_shared<opset5::Reshape>(m_beta, new_shape, true);
        auto mean_aligned = make_shared<opset5::Reshape>(m_mean, new_shape, true);

        // input_sub_mean = input - mean
        auto input_sub_mean = register_new_node<opset5::Subtract>(m_input, mean_aligned);
        // Multiply  `input - mean` and `gamma / sqrt(variance + eps)`
        auto mul = std::make_shared<opset5::Multiply>(input_sub_mean, gamma_div_scale_aligned);
        // Add `(input - mean) * gamma / sqrt(variance + eps)` and `beta`
        auto add = std::make_shared<opset5::Add>(mul, beta_aligned);

        add->set_friendly_name(m_bn->get_friendly_name());

        copy_runtime_info(m_bn, {scale_add, scale, gamma_div_scale, gamma_div_scale_aligned,
            beta_aligned, input_sub_mean, mul, add});

        replace_node(m_bn, add);

        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(bn, "BatchNormDecomposition");
    this->register_matcher(m, callback);
}
