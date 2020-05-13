// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_nonzero.hpp"

#include <common_test_utils/test_common.hpp>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/function.hpp>

#include <details/ie_exception.hpp>

#include <gtest/gtest.h>

namespace {

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::PartialShape;

typedef std::tuple<
    TensorType, // input type
    TensorShape, // input shape
    TensorType // output type
> staticShapeNonZeroTestParams;

class StaticShapeNonZeroTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<staticShapeNonZeroTestParams> {
public:
    void SetUp() override {
        const auto& parameters  = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters);
        m_outputType            = std::get<2>(parameters);

        m_param = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
    }
protected:
    std::shared_ptr<ngraph::opset3::Parameter> m_param;
    ngraph::element::Type m_outputType;
};

std::vector<ngraph::PartialShape> testStaticShapes {
        TensorShape{1000},
        TensorShape{4, 1000},
        TensorShape{3, 128, 256},
        TensorShape{2, 3, 128, 256},
};

std::vector<ngraph::PartialShape> testDynamicShapes {
        TensorShape{ngraph::Dimension::dynamic()},
        TensorShape{4, ngraph::Dimension::dynamic()},
        TensorShape{3, ngraph::Dimension::dynamic(), 256},
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

std::vector<ngraph::element::Type> outputTypes {
        ngraph::element::i32,
        ngraph::element::i64,
};


//
// Positive tests
//

TEST_P(StaticShapeNonZeroTests, CanValidateAndInferTypes) {
    std::shared_ptr<ngraph::vpu::op::StaticShapeNonZero> op;
    ASSERT_NO_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(m_param, m_outputType));
    ASSERT_NO_THROW(std::make_shared<ngraph::Function>(
            ngraph::OutputVector{op->output(0), op->output(1)},
            ngraph::ParameterVector{m_param}));
}

INSTANTIATE_TEST_CASE_P(NGraph, StaticShapeNonZeroTests, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testStaticShapes),
        testing::ValuesIn(outputTypes))
);

//
// Negative tests
//

using StaticShapeNonZeroTestsNegativeInputDataType = StaticShapeNonZeroTests;
TEST_P(StaticShapeNonZeroTestsNegativeInputDataType, ThrowsOnInvalidInputType) {
    std::shared_ptr<ngraph::vpu::op::StaticShapeNonZero> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(m_param, m_outputType),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_CASE_P(NGraph, StaticShapeNonZeroTestsNegativeInputDataType, testing::Combine(
        testing::Values(ngraph::element::boolean),
        testing::ValuesIn(testStaticShapes),
        testing::ValuesIn(outputTypes))
);

using StaticShapeNonZeroTestsNegativeDataShape = StaticShapeNonZeroTests;
TEST_P(StaticShapeNonZeroTestsNegativeDataShape, ThrowsOnInvalidDataShape) {
    std::shared_ptr<ngraph::vpu::op::StaticShapeNonZero> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(m_param, m_outputType),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_CASE_P(NGraph, StaticShapeNonZeroTestsNegativeDataShape, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testDynamicShapes),
        testing::ValuesIn(outputTypes))
);

using StaticShapeNonZeroTestsNegativeOutputDataType = StaticShapeNonZeroTests;
TEST_P(StaticShapeNonZeroTestsNegativeOutputDataType, ThrowsOnInvalidOutputType) {
    std::shared_ptr<ngraph::vpu::op::StaticShapeNonZero> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(m_param, m_outputType),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_CASE_P(NGraph, StaticShapeNonZeroTestsNegativeOutputDataType, testing::Combine(
        testing::ValuesIn(testNGraphNumericTypes),
        testing::ValuesIn(testStaticShapes),
        testing::Values(ngraph::element::boolean))
);

}  // namespace
