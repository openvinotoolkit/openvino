// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"

#include <common_test_utils/test_common.hpp>

#include <ngraph/op/parameter.hpp>
#include <ngraph/function.hpp>


#include <gtest/gtest.h>

namespace {

using TensorShape = ngraph::PartialShape;
using TensorType = ngraph::element::Type;

using TestParams = std::tuple<
        TensorShape,
        TensorType,
        TensorShape,
        TensorType>;

class OutShapeOfReshapeTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<TestParams> {
public:
    void SetUp() override {
        const auto& parameters  = GetParam();
        const auto& inDataShapeTensorShape  = std::get<0>(parameters);
        const auto& inTensorShapeTensorType = std::get<1>(parameters);
        const auto& outShapeDescriptorTensorShape = std::get<2>(parameters);
        const auto& outShapeDescriptorTensorType = std::get<3>(parameters);

        m_inDataShapeParam = std::make_shared<ngraph::op::Parameter>(
                inTensorShapeTensorType, inDataShapeTensorShape);
        m_outShapeDescriptorParam = std::make_shared<ngraph::op::Parameter>(
                outShapeDescriptorTensorType, outShapeDescriptorTensorShape);
    }

protected:
    std::shared_ptr<ngraph::op::Parameter> m_inDataShapeParam;
    std::shared_ptr<ngraph::op::Parameter> m_outShapeDescriptorParam;
};

std::vector<ngraph::PartialShape> tensorShapes {
        TensorShape{1},
        TensorShape{3},
        TensorShape{4},
};

std::set<ngraph::element::Type> allNGraphTypes() {
    return {
            ngraph::element::dynamic,
            ngraph::element::boolean,
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
            ngraph::element::u64
    };
}

std::set<ngraph::element::Type> allNGraphIntegralNumberTypes() {
    return {
            ngraph::element::i8,
            ngraph::element::i16,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u1,
            ngraph::element::u8,
            ngraph::element::u16,
            ngraph::element::u32,
            ngraph::element::u64
    };
}

//
// Positive tests
//

TEST_P(OutShapeOfReshapeTests, CanValidateAndInferTypes) {
    std::shared_ptr<ngraph::vpu::op::OutShapeOfReshape> op;
    ASSERT_NO_THROW(op = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(
            m_inDataShapeParam, m_outShapeDescriptorParam, true));
    ASSERT_NO_THROW(auto fun = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{op->output(0)},
            ngraph::ParameterVector{m_inDataShapeParam, m_outShapeDescriptorParam}));
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, OutShapeOfReshapeTests, testing::Combine(
        testing::ValuesIn(tensorShapes),
        testing::ValuesIn(allNGraphIntegralNumberTypes()),
        testing::ValuesIn(tensorShapes),
        testing::ValuesIn(allNGraphIntegralNumberTypes()))
);

//
// Negative tests
//

std::set<ngraph::element::Type> allNGraphNotIntegralTypes() {
    auto notIntegralTypes = std::set<ngraph::element::Type>{};
    const auto& allTypes = allNGraphTypes();
    const auto& allIntegralTypes = allNGraphIntegralNumberTypes();
    std::set_difference(allTypes.cbegin(), allTypes.cend(), allIntegralTypes.cbegin(), allIntegralTypes.cend(),
                        std::inserter(notIntegralTypes, notIntegralTypes.begin()));
    return notIntegralTypes;
}

using OutShapeOfReshapeTestsNegativeDataType = OutShapeOfReshapeTests;
TEST_P(OutShapeOfReshapeTestsNegativeDataType, ThrowsOnInvalidDataType) {
    std::shared_ptr<ngraph::vpu::op::OutShapeOfReshape> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(
            m_inDataShapeParam, m_outShapeDescriptorParam, true),
                 ngraph::NodeValidationFailure);
}
INSTANTIATE_TEST_SUITE_P(smoke_InvalidInDataShapeTensorType, OutShapeOfReshapeTestsNegativeDataType,
        testing::Combine(
            testing::Values(TensorShape{4}),
            testing::ValuesIn(allNGraphNotIntegralTypes()),
            testing::Values(TensorShape{3}),
            testing::Values(ngraph::element::i64))
);

INSTANTIATE_TEST_SUITE_P(smoke_InvalidOutShapeDescriptorTensorType, OutShapeOfReshapeTestsNegativeDataType,
        testing::Combine(
            testing::Values(TensorShape{4}),
            testing::Values(ngraph::element::i64),
            testing::Values(TensorShape{3}),
            testing::ValuesIn(allNGraphNotIntegralTypes()))
);

std::vector<ngraph::PartialShape> invalidTensorShapes {
        TensorShape{},
        TensorShape{4, 8},
        TensorShape{ngraph::Dimension::dynamic()},
};

using OutShapeOfReshapeTestsNegativeDataShape = OutShapeOfReshapeTests;
TEST_P(OutShapeOfReshapeTestsNegativeDataShape, ThrowsOnInvalidDataShape) {
    std::shared_ptr<ngraph::vpu::op::OutShapeOfReshape> op;
    ASSERT_THROW(op = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(
            m_inDataShapeParam, m_outShapeDescriptorParam, true),
                 ngraph::NodeValidationFailure);
}

INSTANTIATE_TEST_SUITE_P(smoke_InvalidInDataShapeTensorShape, OutShapeOfReshapeTestsNegativeDataShape,
        testing::Combine(
            testing::ValuesIn(invalidTensorShapes),
            testing::Values(ngraph::element::i64),
            testing::ValuesIn(tensorShapes),
            testing::Values(ngraph::element::i64))
);

INSTANTIATE_TEST_SUITE_P(smoke_InvalidOutShapeDescriptorTensorShape, OutShapeOfReshapeTestsNegativeDataShape,
        testing::Combine(
            testing::ValuesIn(tensorShapes),
            testing::Values(ngraph::element::i64),
            testing::ValuesIn(invalidTensorShapes),
            testing::Values(ngraph::element::i64))
);

}  // namespace
