// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/batch_norm_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset5.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

std::shared_ptr<ov::Model> get_ref_model_with_dyn_shapes(ov::element::Type precision, const PartialShape& input_shape) {
    auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
    auto gamma = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
    auto beta = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
    auto mean = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
    auto var = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
    // scale_add = variance + eps
    auto scale_add = std::make_shared<ov::op::v1::Add>(var, ov::op::v0::Constant::create(precision, Shape{}, {0.001}));
    // scale = sqrt(variance + eps)
    auto scale = std::make_shared<ov::op::v0::Sqrt>(scale_add);
    // Divide `gamma` by `sqrt(variance + eps)`
    auto gamma_div_scale = std::make_shared<ov::op::v1::Divide>(gamma, scale);

    int64_t dims_to_add = input->get_partial_shape().rank().get_length() - 2;
    const auto one = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    const auto tail_shape_rank = ov::op::v0::Constant::create(element::i64, Shape{1}, {dims_to_add});
    const auto tail_shape = std::make_shared<ov::op::v3::Broadcast>(one, tail_shape_rank);
    const auto C_dim = std::make_shared<ov::op::v3::ShapeOf>(gamma);
    // create new shape [1, C, 1, 1, ...]
    const auto new_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{one, C_dim, tail_shape}, 0);

    std::shared_ptr<Node> gamma_div_scale_aligned =
        std::make_shared<ov::op::v1::Reshape>(gamma_div_scale, new_shape, true);
    std::shared_ptr<Node> beta_aligned = std::make_shared<ov::op::v1::Reshape>(beta, new_shape, true);
    std::shared_ptr<Node> mean_aligned = std::make_shared<ov::op::v1::Reshape>(mean, new_shape, true);
    std::shared_ptr<Node> mean_negative = std::make_shared<ov::op::v1::Multiply>(
        mean_aligned,
        ov::op::v0::Constant::create(mean_aligned->get_output_element_type(0), Shape{}, {-1}));

    // input_sub_mean = input + mean * -1
    auto input_sub_mean = std::make_shared<ov::op::v1::Add>(input, mean_negative);
    // Multiply  `input - mean` and `gamma / sqrt(variance + eps)`
    auto mul = std::make_shared<ov::op::v1::Multiply>(input_sub_mean, gamma_div_scale_aligned);
    // Add `(input - mean) * gamma / sqrt(variance + eps)` and `beta`
    auto add = std::make_shared<ov::op::v1::Add>(mul, beta_aligned);

    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input, gamma, beta, mean, var});
}

TEST_F(TransformationTestsF, BatchNormDecompositionStaticRankOpset1) {
    const PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = element::f32;
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto gamma = opset1::Constant::create(precision, Shape{3}, {3});
        auto beta = opset1::Constant::create(precision, Shape{3}, {3});
        auto mean = opset1::Constant::create(precision, Shape{3}, {3});
        auto var = opset1::Constant::create(precision, Shape{3}, {3});
        auto batch_norm = std::make_shared<opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        model = std::make_shared<ov::Model>(NodeVector{batch_norm}, ParameterVector{input});
        manager.register_pass<ov::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto add_const_1 = opset1::Constant::create(precision, {1, 3, 1, 1}, {-3});
        auto add_1 = std::make_shared<opset1::Add>(input, add_const_1);
        auto mul_const = opset1::Constant::create(precision, {1, 3, 1, 1}, {1.7317622900009155});
        auto mul = std::make_shared<opset1::Multiply>(add_1, mul_const);
        auto add_const_2 = opset1::Constant::create(precision, {1, 3, 1, 1}, {3});
        auto add_2 = std::make_shared<opset1::Add>(mul, add_const_2);

        model_ref = std::make_shared<ov::Model>(NodeVector{add_2}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, BatchNormDecompositionStaticRankOpset5) {
    const PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = element::f32;
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto gamma = opset1::Constant::create(precision, Shape{3}, {3});
        auto beta = opset1::Constant::create(precision, Shape{3}, {3});
        auto mean = opset1::Constant::create(precision, Shape{3}, {3});
        auto var = opset1::Constant::create(precision, Shape{3}, {3});
        auto batch_norm = std::make_shared<opset5::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        model = std::make_shared<ov::Model>(NodeVector{batch_norm}, ParameterVector{input});
        manager.register_pass<ov::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto add_const_1 = opset1::Constant::create(precision, {1, 3, 1, 1}, {-3});
        auto add_1 = std::make_shared<opset1::Add>(input, add_const_1);
        auto mul_const = opset1::Constant::create(precision, {1, 3, 1, 1}, {1.7317622900009155});
        auto mul = std::make_shared<opset1::Multiply>(add_1, mul_const);
        auto add_const_2 = opset1::Constant::create(precision, {1, 3, 1, 1}, {3});
        auto add_2 = std::make_shared<opset1::Add>(mul, add_const_2);

        model_ref = std::make_shared<ov::Model>(NodeVector{add_2}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, BatchNormDecompositionDynamicShapesOpset1) {
    const PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = element::f32;
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto gamma = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto beta = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto mean = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto var = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto batch_norm = std::make_shared<opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        model = std::make_shared<ov::Model>(NodeVector{batch_norm}, ParameterVector{input, gamma, beta, mean, var});
        manager.register_pass<ov::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    { model_ref = get_ref_model_with_dyn_shapes(precision, input_shape); }
}

TEST_F(TransformationTestsF, BatchNormDecompositionDynamicShapesOpset5) {
    const PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = element::f32;
    {
        auto input = std::make_shared<opset1::Parameter>(precision, input_shape);
        auto gamma = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto beta = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto mean = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto var = std::make_shared<opset1::Parameter>(precision, PartialShape{-1});
        auto batch_norm = std::make_shared<opset5::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        model = std::make_shared<ov::Model>(NodeVector{batch_norm}, ParameterVector{input, gamma, beta, mean, var});
        manager.register_pass<ov::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    { model_ref = get_ref_model_with_dyn_shapes(precision, input_shape); }
}

TEST_F(TransformationTestsF, BatchNormDecompositionDynamicRank) {
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto gamma = opset1::Constant::create(element::f32, Shape{3}, {3});
        auto beta = opset1::Constant::create(element::f32, Shape{3}, {3});
        auto mean = opset1::Constant::create(element::f32, Shape{3}, {3});
        auto var = opset1::Constant::create(element::f32, Shape{3}, {3});
        auto broadcast = std::make_shared<opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);
        broadcast->set_friendly_name("broadcast");

        model = std::make_shared<ov::Model>(NodeVector{broadcast}, ParameterVector{input});
        manager.register_pass<ov::pass::BatchNormDecomposition>();
    }
}
