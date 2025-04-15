// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>

#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::builder::subgraph;

TEST(LPT, isConstantPathFQAfterInputTransformation) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto fqOnActivations = makeFakeQuantize(input, ov::element::f32,
        FakeQuantizeOnData{ 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(fqOnActivations);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathFQAfterWeightsTransformation) {
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto fqOnWeights = makeFakeQuantize(weights, ov::element::f32,
        FakeQuantizeOnWeights{ 255ul, {}, {0.f}, {254.f}, {-1.27f}, {1.27f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(fqOnWeights);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathDqAfterInputTransformation) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto dqOnActivations = makeDequantization(input, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(dqOnActivations);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathDqAfterWeightsTransformation) {
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto dqOnWeights = makeDequantization(weights, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(dqOnWeights);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathTwoInputsTransformation) {
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ov::opset1::MatMul>(dq1, dq2);

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(matmul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathTwoConsantsTransformation) {
    const auto constant1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto constant2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 3, 1, 1, 1 }, { 1.f });
    const auto dq1 = makeDequantization(constant1, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(constant2, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto eltwise = std::make_shared<ov::opset1::Add>(dq1, dq2);

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(eltwise);

    ASSERT_EQ(true, result);
}

TEST(LPT, isConstantPathMatMulParentFQTransformation) {
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ov::opset1::MatMul>(dq1, dq2);
    const auto fqAfterMatMul = makeFakeQuantize(matmul, ov::element::f32,
        FakeQuantizeOnWeights{ 255ul, {}, {0.f}, {254.f}, {-1.27f}, {1.27f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(fqAfterMatMul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathMatMulParentDqTransformation) {
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto dq1 = makeDequantization(input1, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto dq2 = makeDequantization(input2, DequantizationOperations{ ov::element::f32, {128.f}, {0.1f} });
    const auto matmul = std::make_shared<ov::opset1::MatMul>(dq1, dq2);
    const auto dqAfterMatMul = makeDequantization(matmul, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(dqAfterMatMul);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathConvParentDqTransformation) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 72, 16 });
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 6, 3, 1, 1 }, { 1.f });
    const auto conv = std::make_shared<ov::opset1::Convolution>(
        input,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    const auto dqAfterConv = makeDequantization(conv, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(dqAfterConv);

    ASSERT_EQ(false, result);
}

TEST(LPT, isConstantPathGroupConvParentDqTransformation) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 6, 3, 1, 1 }, { 1.f });
    const auto groupConv = std::make_shared<ov::opset1::GroupConvolution>(
        input,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    const auto dqAfterGroupConv = makeDequantization(groupConv, DequantizationOperations{ {}, {}, {0.1f} });

    const bool result = ov::pass::low_precision::NetworkHelper::isConstantPath(dqAfterGroupConv);

    ASSERT_EQ(false, result);
}
