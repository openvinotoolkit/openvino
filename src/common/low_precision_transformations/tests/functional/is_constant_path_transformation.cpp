// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>

#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

TEST(LPT, isConstantPathFQAfterInputTransformation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto fqOnActivations = makeFakeQuantize(input, ngraph::element::f32,
        FakeQuantizeOnData{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(fqOnActivations);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathFQAfterWeightsTransformation) {
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto fqOnWeights = makeFakeQuantize(weights, ngraph::element::f32,
        FakeQuantizeOnWeights{ 255ul, {}, {0.f}, {254.f}, {-1.27f}, {1.27f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(fqOnWeights);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathDqAfterInputTransformation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto dqOnActivations = makeDequantization(input, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(dqOnActivations);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathDqAfterWeightsTransformation) {
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto dqOnWeights = makeDequantization(weights, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(dqOnWeights);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathTwoInputsTransformation) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ngraph::opset1::MatMul>(dq1, dq2);

    const bool result = low_precision::NetworkHelper::isConstantPath(matmul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathTwoConsantsTransformation) {
    const auto constant1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto constant2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto dq1 = makeDequantization(constant1, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(constant2, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto eltwise = std::make_shared<ngraph::opset1::Add>(dq1, dq2);

    const bool result = low_precision::NetworkHelper::isConstantPath(eltwise);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathMatMulParentFQTransformation) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ngraph::opset1::MatMul>(dq1, dq2);
    const auto fqAfterMatMul = makeFakeQuantize(matmul, ngraph::element::f32,
        FakeQuantizeOnWeights{ 255ul, {}, {0.f}, {254.f}, {-1.27f}, {1.27f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(fqAfterMatMul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathMatMulParentDqTransformation) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ngraph::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ngraph::opset1::MatMul>(dq1, dq2);
    const auto dqAfterMatMul = makeDequantization(matmul, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(dqAfterMatMul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathConvParentDqTransformation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 72, 16 });
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 1, 1 }, { 1.f });
    const auto conv = std::make_shared<ngraph::opset1::Convolution>(
        input,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    const auto dqAfterConv = makeDequantization(conv, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(dqAfterConv);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathGroupConvParentDqTransformation) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 6, 3, 1, 1 }, { 1.f });
    const auto groupConv = std::make_shared<ngraph::opset1::GroupConvolution>(
        input,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    const auto dqAfterGroupConv = makeDequantization(groupConv, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = low_precision::NetworkHelper::isConstantPath(dqAfterGroupConv);

    ASSERT_EQ(false, result);
}
