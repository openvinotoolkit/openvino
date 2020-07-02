// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/operations/out_shape_of_reshape.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/function.hpp"

#include "common_test_utils/test_common.hpp"
#include "gtest/gtest.h"

#include <string>
#include <memory>
#include <map>
#include <vector>

namespace {

using DataType  = ngraph::element::Type;
using DataShape = ngraph::Shape;
using TestParams = std::tuple<DataShape, DataType>;

class DynamicToStaticShapeReshapeTests : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<TestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& inDataShape  = std::get<0>(parameters);
        const auto& inDataType = std::get<1>(parameters);

        ngraph::helpers::CompareFunctions(*transform(inDataType, inDataShape), *reference(inDataType, inDataShape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(const ngraph::element::Type& inDataType, const ngraph::Shape& inDataShape) const {
        const auto inDataParam = std::make_shared<ngraph::op::Parameter>(inDataType, inDataShape);
        const auto inDataDimsParam = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64, ngraph::Shape{inDataShape.size()});
        const auto outShapeDescriptorParam = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{inDataShape.size()}, inDataShape);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(inDataParam, inDataDimsParam);
        const auto reshape = std::make_shared<ngraph::op::v1::Reshape>(dsr, outShapeDescriptorParam, true);

        auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{reshape},
            ngraph::ParameterVector{inDataParam, inDataDimsParam},
            "Actual");
        reshape->set_output_type(
            0,
            dsr->get_input_element_type(0),
            ngraph::PartialShape::dynamic(outShapeDescriptorParam->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{ngraph::op::v1::Reshape::type_info, vpu::dynamicToStaticShapeReshape}};
        vpu::DynamicToStaticShape(transformations).transform(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(const ngraph::element::Type& inDataType, const ngraph::Shape& inDataShape) const {
        const auto inDataParam = std::make_shared<ngraph::op::Parameter>(inDataType, inDataShape);
        const auto inDataDimsParam = std::make_shared<ngraph::op::Parameter>(ngraph::element::i64, ngraph::Shape{inDataShape.size()});
        const auto outShapeDescriptorParam = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{inDataShape.size()}, inDataShape);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(inDataParam, inDataDimsParam);
        const auto reshape = std::make_shared<ngraph::op::v1::Reshape>(dsr0, outShapeDescriptorParam, true);

        const auto outShapeOfReshape = std::make_shared<ngraph::vpu::op::OutShapeOfReshape>(inDataDimsParam, outShapeDescriptorParam, true);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(reshape, outShapeOfReshape);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr1},
            ngraph::ParameterVector{inDataParam, inDataDimsParam},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeReshapeTests, compareFunctions) {}

INSTANTIATE_TEST_CASE_P(NGraph, DynamicToStaticShapeReshapeTests, testing::Combine(
    testing::Values(
        DataShape{4, 1000},
        DataShape{3, 128, 256},
        DataShape{2, 3, 128, 256}),
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8)
));

}  // namespace
