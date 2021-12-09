// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"

#include <common_test_utils/test_common.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/function.hpp>


#include <gtest/gtest.h>

namespace {

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::PartialShape;
using AxesMapping = std::vector<size_t>;

struct BroadcastNumpyShapes {
    TensorShape srcShape;
    TensorShape targetShape;
};

struct BroadcastExplicitShapes {
    TensorShape srcShape;
    TensorShape targetShape;
    AxesMapping axesMapping;
};

struct BroadcastBidirectionalShapes {
    TensorShape srcShape;
    TensorShape targetShape;
    TensorShape outputShape;
};
using BroadcastNumpyTestParams = std::tuple<TensorType, BroadcastNumpyShapes>;
using BroadcastExplicitTestParams = std::tuple<TensorType, BroadcastExplicitShapes>;
using BroadcastBidirectionalTestParams = std::tuple<TensorType, BroadcastBidirectionalShapes>;

class StaticShapeBroadcastNumpyTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<BroadcastNumpyTestParams> {
public:
    void SetUp() override {
        const auto& parameters  = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters).srcShape;
        const auto& targetShape = std::get<1>(parameters).targetShape;

        m_tensor = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        m_tensorWithTargetShape = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);
    }
protected:
    std::shared_ptr<ngraph::opset3::Parameter> m_tensor;
    std::shared_ptr<ngraph::opset3::Parameter> m_tensorWithTargetShape;
};

class StaticShapeBroadcastExplicitTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<BroadcastExplicitTestParams> {
public:
    void SetUp() override {
        const auto& parameters  = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters).srcShape;
        const auto& targetShape = std::get<1>(parameters).targetShape;
        const auto& axesMapping = std::get<1>(parameters).axesMapping;

        m_tensor = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        m_tensorWithTargetShape = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);
        m_axesMapping = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::u64, ngraph::Shape{axesMapping.size()}, axesMapping);
    }
protected:
    std::shared_ptr<ngraph::opset3::Parameter> m_tensor;
    std::shared_ptr<ngraph::opset3::Parameter> m_tensorWithTargetShape;
    std::shared_ptr<ngraph::opset3::Constant> m_axesMapping;
};

class StaticShapeBroadcastBidirectionalTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<BroadcastBidirectionalTestParams> {
public:
    void SetUp() override {
        const auto& parameters  = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters).srcShape;
        const auto& targetShape = std::get<1>(parameters).targetShape;
        const auto& outputShape = std::get<1>(parameters).outputShape;

        m_tensor = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        m_tensorWithTargetShape = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);
        m_tensorWithOutput = std::make_shared<ngraph::opset3::Parameter>(tensorType, outputShape);
    }
protected:
    std::shared_ptr<ngraph::opset3::Parameter> m_tensor;
    std::shared_ptr<ngraph::opset3::Parameter> m_tensorWithTargetShape;
    std::shared_ptr<ngraph::opset3::Parameter> m_tensorWithOutput;
};

std::vector<BroadcastNumpyShapes> testNumpyStaticShapes {
        BroadcastNumpyShapes{TensorShape{1, 100}, TensorShape{4, 100}},
        BroadcastNumpyShapes{TensorShape{1, 100}, TensorShape{2, 4, 100}},
        BroadcastNumpyShapes{TensorShape{16, 1, 1}, TensorShape{2, 16, 50, 50}},
};

std::vector<BroadcastExplicitShapes> testExplicitStaticShapes {
        BroadcastExplicitShapes{TensorShape{16}, TensorShape{1, 16, 50, 50}, AxesMapping{1}},
        BroadcastExplicitShapes{TensorShape{50, 50}, TensorShape{1, 50, 50, 16}, AxesMapping{1, 2}},
};

std::vector<BroadcastBidirectionalShapes> testBidirectionalStaticShapes {
        BroadcastBidirectionalShapes{TensorShape{1, 100}, TensorShape{4, 100}, TensorShape{4, 100}},
        BroadcastBidirectionalShapes{TensorShape{1, 100}, TensorShape{2, 4, 100}, TensorShape{2, 4, 100}},
        BroadcastBidirectionalShapes{TensorShape{16, 1, 1}, TensorShape{2, 16, 50, 50}, TensorShape{2, 16, 50, 50}},
        BroadcastBidirectionalShapes{TensorShape{4, 100}, TensorShape{1, 100}, TensorShape{4, 100}},
        BroadcastBidirectionalShapes{TensorShape{2, 4, 100}, TensorShape{1, 100}, {2, 4, 100}},
        BroadcastBidirectionalShapes{TensorShape{2, 16, 1, 50}, TensorShape{16, 50, 1}, TensorShape{2, 16, 50, 50}},
};

std::vector<ngraph::element::Type> testNGraphNumericTypes {
        ngraph::element::dynamic,
        ngraph::element::bf16,
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::f64,
        ngraph::element::i8,
        ngraph::element::i16,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u1,
        ngraph::element::u8,
        ngraph::element::u16,
        ngraph::element::u32,
        ngraph::element::u64,
};

//
// Positive tests
//

TEST_P(StaticShapeBroadcastNumpyTests, CanValidateAndInferTypes) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_NO_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf));
    ASSERT_NO_THROW(auto fun = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{op->output(0)},
            ngraph::ParameterVector{m_tensor, m_tensorWithTargetShape}));
    ASSERT_EQ(m_tensorWithTargetShape->get_shape(), op->output(0).get_shape());
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastNumpyTests, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testNumpyStaticShapes))
);

TEST_P(StaticShapeBroadcastExplicitTests, CanValidateAndInferTypes) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_NO_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf, m_axesMapping));
    ASSERT_NO_THROW(auto fun = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{op->output(0)},
            ngraph::ParameterVector{m_tensor, m_tensorWithTargetShape}));
    ASSERT_EQ(m_tensorWithTargetShape->get_shape(), op->get_output_shape(0));
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastExplicitTests, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testExplicitStaticShapes))
);

TEST_P(StaticShapeBroadcastBidirectionalTests, CanValidateAndInferTypes) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_NO_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf, ngraph::op::BroadcastType::BIDIRECTIONAL));
    ASSERT_NO_THROW(auto fun = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{op->output(0)},
            ngraph::ParameterVector{m_tensor, m_tensorWithTargetShape}));
    ASSERT_EQ(m_tensorWithOutput->get_shape(), op->output(0).get_shape());
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastBidirectionalTests, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testBidirectionalStaticShapes))
);


//
// Negative tests
//

using StaticShapeBroadcastNumpyTestsNegativeNumInputs = StaticShapeBroadcastNumpyTests;
TEST_P(StaticShapeBroadcastNumpyTestsNegativeNumInputs, ThrowsOnInvalidNumInputs) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    const auto axesMapping = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64, ngraph::Shape{1}, 0);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf, axesMapping, ngraph::op::BroadcastType::NUMPY),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastNumpyTestsNegativeNumInputs, testing::Combine(
        testing::Values(ngraph::element::f16),
        testing::Values(testNumpyStaticShapes[0]))
);

using StaticShapeBroadcastExplicitTestsNegativeNumInputs = StaticShapeBroadcastExplicitTests;
TEST_P(StaticShapeBroadcastExplicitTestsNegativeNumInputs, ThrowsOnInvalidNumInputs) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf, ngraph::op::BroadcastType::EXPLICIT),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastExplicitTestsNegativeNumInputs, testing::Combine(
        testing::Values(ngraph::element::f16),
        testing::Values(testExplicitStaticShapes[0]))
);

using StaticShapeBroadcastBidirectionalTestsNegativeNumInputs = StaticShapeBroadcastBidirectionalTests;
TEST_P(StaticShapeBroadcastBidirectionalTestsNegativeNumInputs, ThrowsOnInvalidNumInputs) {
    const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(m_tensorWithTargetShape);
    const auto axesMapping = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64, ngraph::Shape{1}, 0);
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, shapeOf, axesMapping, ngraph::op::BroadcastType::BIDIRECTIONAL),
                 ngraph::NodeValidationFailure);
}

using StaticShapeBroadcastTestsNegativeMode = StaticShapeBroadcastNumpyTests;
INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastTestsNegativeMode, testing::Combine(
        testing::Values(ngraph::element::f16),
        testing::Values(testNumpyStaticShapes[0]))
);

using StaticShapeBroadcastTestsNegativeEvaluate = StaticShapeBroadcastNumpyTests;
TEST_P(StaticShapeBroadcastTestsNegativeEvaluate, ThrowsOnInvalidMode) {
    const auto targetShape = std::make_shared<ngraph::opset3::Parameter>(
            ngraph::element::u64, ngraph::Shape{4});
    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            m_tensor, targetShape), ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, StaticShapeBroadcastTestsNegativeEvaluate, testing::Combine(
        testing::Values(ngraph::element::f16),
        testing::Values(testNumpyStaticShapes[0]))
);

}  // namespace
