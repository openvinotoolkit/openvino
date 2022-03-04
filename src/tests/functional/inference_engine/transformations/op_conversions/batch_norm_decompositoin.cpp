// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, BatchNormDecompositionStaticRankOpset1) {
    const ngraph::PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = ngraph::element::f32;
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(precision, input_shape);
        auto gamma = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto beta = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto mean = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto var = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto batch_norm = std::make_shared<ngraph::opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_norm}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(precision, input_shape);
        auto add_const_1 = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {-3});
        auto add_1 = std::make_shared<ngraph::opset1::Add>(input, add_const_1);
        auto mul_const = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {1.7317622900009155});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(add_1, mul_const);
        auto add_const_2 = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {3});
        auto add_2 = std::make_shared<ngraph::opset1::Add>(mul, add_const_2);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add_2}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, BatchNormDecompositionStaticRankOpset5) {
    const ngraph::PartialShape input_shape{-1, -1, -1, -1};
    const auto precision = ngraph::element::f32;
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(precision, input_shape);
        auto gamma = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto beta = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto mean = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto var = ngraph::opset1::Constant::create(precision, ngraph::Shape{3}, {3});
        auto batch_norm = std::make_shared<ngraph::opset5::BatchNormInference>(input, gamma, beta, mean, var, 0.001);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{batch_norm}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::BatchNormDecomposition>();
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(precision, input_shape);
        auto add_const_1 = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {-3});
        auto add_1 = std::make_shared<ngraph::opset1::Add>(input, add_const_1);
        auto mul_const = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {1.7317622900009155});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(add_1, mul_const);
        auto add_const_2 = ngraph::opset1::Constant::create(precision, {1, 3, 1, 1}, {3});
        auto add_2 = std::make_shared<ngraph::opset1::Add>(mul, add_const_2);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add_2}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, BatchNormDecompositionDynamicRank) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto gamma = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto beta = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto mean = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto var = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto broadcast = std::make_shared<ngraph::opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);
        broadcast->set_friendly_name("broadcast");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::BatchNormDecomposition>();
    }
}