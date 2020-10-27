// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/batch_norm_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::BatchNormDecomposition, "BatchNormDecomposition", 0);

ngraph::pass::BatchNormDecomposition::BatchNormDecomposition() {
    auto bn = pattern::wrap_type<opset1::BatchNormInference>({
        pattern::any_input(pattern::has_static_rank()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape())
    });

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto m_bn = dynamic_pointer_cast<opset1::BatchNormInference>(m.get_match_root());
        if (!m_bn) {
            return false;
        }

        auto m_gamma = m_bn->input_value(0);
        auto m_beta = m_bn->input_value(1);
        auto m_input = m_bn->input_value(2);
        auto m_mean = m_bn->input_value(3);
        auto m_var = m_bn->input_value(4);

        const auto& input_type = m_input.get_element_type();
        // scale_add = variance + eps
        auto scale_add = make_shared<opset5::Add>(m_var, opset5::Constant::create(input_type, Shape{}, {m_bn->get_eps_value()}));
        // scale = sqrt(variance + eps)
        auto scale = make_shared<opset5::Sqrt>(scale_add);
        // Divide `gamma` by `sqrt(variance + eps)`
        auto gamma_div_scale = std::make_shared<opset5::Divide>(m_gamma, scale);

        int64_t dims_to_add = m_input.get_partial_shape().rank().get_length() - 2;

        // TODO: instead of getting full shape we can concatenate sequence of ones with ShapeOf
        Shape input_aligned_shape = m_gamma.get_shape();
        for (int64_t i = 0; i < dims_to_add; ++i)
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

// TODO: this pass will be unified with BatchNormDecomposition pass
ngraph::pass::BatchNormV5Decomposition::BatchNormV5Decomposition() {
    auto bn = pattern::wrap_type<opset5::BatchNormInference>({
        pattern::any_input(pattern::has_static_rank()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape()),
        pattern::any_input(pattern::has_static_shape())
    });

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto m_bn = dynamic_pointer_cast<opset5::BatchNormInference>(m.get_match_root());
        if (!m_bn) {
            return false;
        }

        auto m_input = m_bn->input_value(0);
        auto m_gamma = m_bn->input_value(1);
        auto m_beta = m_bn->input_value(2);
        auto m_mean = m_bn->input_value(3);
        auto m_var = m_bn->input_value(4);

        const auto& input_type = m_input.get_element_type();
        // scale_add = variance + eps
        auto scale_add = make_shared<opset5::Add>(m_var, opset5::Constant::create(input_type, Shape{}, {m_bn->get_eps_value()}));
        // scale = sqrt(variance + eps)
        auto scale = make_shared<opset5::Sqrt>(scale_add);
        // Divide `gamma` by `sqrt(variance + eps)`
        auto gamma_div_scale = std::make_shared<opset5::Divide>(m_gamma, scale);

        int64_t dims_to_add = m_input.get_partial_shape().rank().get_length() - 2;

        // TODO: instead of getting full shape we can concatenate sequence of ones with ShapeOf
        Shape input_aligned_shape = m_gamma.get_shape();
        for (int64_t i = 0; i < dims_to_add; ++i)
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
