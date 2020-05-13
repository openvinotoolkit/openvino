// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataShape = ngraph::Shape;
using ShapeDescriptor = std::vector<int32_t>;
using ReshapeTestParams = std::tuple<DataShape, bool, ShapeDescriptor>;

using Parameters = std::tuple<
        DataType,
        ReshapeTestParams,
        LayerTestsUtils::TargetDevice
>;

class DSR_Reshape : public testing::WithParamInterface<Parameters>, public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(GetParam());
        const auto& reshapeTestParams = std::get<1>(GetParam());
        targetDevice = std::get<2>(GetParam());

        const auto& inDataShape = std::get<0>(reshapeTestParams);
        const auto& specialZero = std::get<1>(reshapeTestParams);
        const auto& outShapeDescriptor = std::get<2>(reshapeTestParams);

        const auto inDataParam = std::make_shared<ngraph::op::Parameter>(
                inDataType, inDataShape);
        const auto inDataShapeParam = std::make_shared<ngraph::op::Parameter>(
                ngraph::element::i32, ngraph::Shape{inDataShape.size()});
        const auto dsr  = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
                inDataParam, inDataShapeParam);

        const auto outShapeDescriptorConstNode = std::make_shared<ngraph::op::Constant>(
                ngraph::element::i64, ngraph::Shape{outShapeDescriptor.size()}, outShapeDescriptor);
        const auto reshape = std::make_shared<ngraph::op::v1::Reshape>(
                dsr, outShapeDescriptorConstNode, specialZero);

        const auto result = std::make_shared<ngraph::op::Result>(reshape);
        function = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{result},
                ngraph::ParameterVector{inDataParam, inDataShapeParam},
                "DSR-Reshape");
    }
};

TEST_P(DSR_Reshape, CompareWithReference) {
    Run();
}

std::vector<ReshapeTestParams> reshapeTestParams = {
        std::make_tuple(DataShape{1, 5, 5, 24}, true, ShapeDescriptor{0, -1, 4}),
        std::make_tuple(DataShape{1, 5, 5, 0}, false, ShapeDescriptor{0, 4}),
        std::make_tuple(DataShape{1, 3, 128, 256}, true, ShapeDescriptor{0, 0, 64, 512}),
};

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicReshape, DSR_Reshape,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
                                ::testing::ValuesIn(reshapeTestParams),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
