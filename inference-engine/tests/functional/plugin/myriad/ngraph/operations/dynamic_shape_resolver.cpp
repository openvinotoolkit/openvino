// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/parameter.hpp>
#include <ngraph/function.hpp>

#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>
#include <details/ie_exception.hpp>

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

namespace {

using DataType  = ngraph::element::Type_t;
using DimsType  = ngraph::element::Type_t;
using DataShape = ngraph::Shape;

class DynamicShapeResolverTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<DataType, DimsType, DataShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType   = std::get<0>(parameters);
        const auto& dimsType   = std::get<1>(parameters);
        const auto& dataShape  = std::get<2>(parameters);

        data = std::make_shared<ngraph::op::Parameter>(dataType, dataShape);
        dims = std::make_shared<ngraph::op::Parameter>(dimsType, ngraph::Shape{dataShape.size()});
    }

protected:
    std::shared_ptr<ngraph::op::Parameter> data;
    std::shared_ptr<ngraph::op::Parameter> dims;
};

TEST_P(DynamicShapeResolverTests, CanValidateAndInferTypes) {
    std::shared_ptr<ngraph::op::DynamicShapeResolver> dynamicShapeResolver;
    ASSERT_NO_THROW(dynamicShapeResolver = std::make_shared<ngraph::op::DynamicShapeResolver>(data, dims));
    ASSERT_NO_THROW(std::make_shared<ngraph::Function>(ngraph::NodeVector{dynamicShapeResolver}, ngraph::ParameterVector{data, dims}));
}

std::set<ngraph::element::Type_t> allNGraphTypes() {
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

std::set<ngraph::element::Type_t> allNGraphIntegralNumberTypes() {
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

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverTests, testing::Combine(
    testing::ValuesIn(allNGraphTypes()),
    testing::ValuesIn(allNGraphIntegralNumberTypes()),
    testing::Values(DataShape{1, 800}, DataShape{1, 1})));


using DataPartialShape = ngraph::PartialShape;
using DimsPartialShape = ngraph::PartialShape;
class DynamicShapeResolverNegativeTests
    : public CommonTestUtils::TestsCommon
    , public testing::WithParamInterface<std::tuple<DataType, DimsType, DataPartialShape, DimsPartialShape>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType   = std::get<0>(parameters);
        const auto& dimsType   = std::get<1>(parameters);
        const auto& dataPartialShape  = std::get<2>(parameters);
        const auto& dimsPartialShape  = std::get<3>(parameters);

        data = std::make_shared<ngraph::op::Parameter>(dataType, dataPartialShape);
        dims = std::make_shared<ngraph::op::Parameter>(dimsType, dimsPartialShape);
    }

protected:
    std::shared_ptr<ngraph::op::Parameter> data;
    std::shared_ptr<ngraph::op::Parameter> dims;
};

class DynamicShapeResolverNegativeTestsDimsType : public DynamicShapeResolverNegativeTests {};
TEST_P(DynamicShapeResolverNegativeTestsDimsType, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(std::make_shared<ngraph::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

std::set<ngraph::element::Type_t> allNGraphNotIntegralTypes() {
    auto notIntegralTypes = std::set<ngraph::element::Type_t>{};
    const auto& allTypes = allNGraphTypes();
    const auto& allIntegralTypes = allNGraphIntegralNumberTypes();
    std::set_difference(allTypes.cbegin(), allTypes.cend(), allIntegralTypes.cbegin(), allIntegralTypes.cend(),
        std::inserter(notIntegralTypes, notIntegralTypes.begin()));
    return notIntegralTypes;
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDimsType, testing::Combine(
    testing::ValuesIn(allNGraphTypes()),
    testing::ValuesIn(allNGraphNotIntegralTypes()),
    testing::Values(DataPartialShape{1, 800}),
    testing::Values(DataPartialShape{2})));

class DynamicShapeResolverNegativeTestsDataShape : public DynamicShapeResolverNegativeTests {};
TEST_P(DynamicShapeResolverNegativeTestsDataShape, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(std::make_shared<ngraph::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDataShape, testing::Combine(
    testing::ValuesIn(allNGraphTypes()),
    testing::ValuesIn(allNGraphIntegralNumberTypes()),
    testing::Values(
        DataPartialShape::dynamic(),
        DataPartialShape{{1, ngraph::Dimension::dynamic()}},
        DataPartialShape{{ngraph::Dimension::dynamic(), 1}},
        DataPartialShape{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()}}),
    testing::Values(DataShape{2})));

class DynamicShapeResolverNegativeTestsDimsShape : public DynamicShapeResolverNegativeTests {};
TEST_P(DynamicShapeResolverNegativeTestsDimsShape, ThrowsOnInvalidDimsType) {
    ASSERT_THROW(std::make_shared<ngraph::op::DynamicShapeResolver>(data, dims), ngraph::ngraph_error);
}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicShapeResolverNegativeTestsDimsShape, testing::Combine(
    testing::ValuesIn(allNGraphTypes()),
    testing::ValuesIn(allNGraphIntegralNumberTypes()),
    testing::Values(DataShape{1, 800}),
    testing::Values(
        DataPartialShape::dynamic(),
        DataPartialShape{{1, ngraph::Dimension::dynamic()}},
        DataPartialShape{{ngraph::Dimension::dynamic(), 1}},
        DataPartialShape{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()}},
        DataPartialShape{0},
        DataPartialShape{1},
        DataPartialShape{3})));

}  // namespace
