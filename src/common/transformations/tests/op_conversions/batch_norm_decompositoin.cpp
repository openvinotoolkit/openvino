// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset5.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

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
