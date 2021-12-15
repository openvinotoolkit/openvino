// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

using ShapeDescriptor = std::vector<int32_t>;
using ReshapeTestParams = std::tuple<DataShapeWithUpperBound, bool, ShapeDescriptor>;

using Parameters = std::tuple<
        DataType,
        ReshapeTestParams,
        LayerTestsUtils::TargetDevice>;

class DSR_ReshapeWithStaticDescriptor : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& reshapeTestParams = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);

        const auto& inDataShapes = std::get<0>(reshapeTestParams);
        const auto& specialZero = std::get<1>(reshapeTestParams);
        const auto& outShapeDescriptor = std::get<2>(reshapeTestParams);

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto outShapeDescriptorConstNode = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{outShapeDescriptor.size()}, outShapeDescriptor);

        return std::make_shared<ngraph::opset3::Reshape>(
                inputSubgraph, outShapeDescriptorConstNode, specialZero);
    }
};

TEST_P(DSR_ReshapeWithStaticDescriptor, CompareWithReference) {
    Run();
}

class DSR_ReshapeWithDynamicDescriptor : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<0>(std::get<1>(parameters));
        targetDevice = std::get<2>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto shapeDataType = inputSubgraph->get_input_element_type(1);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(inputSubgraph, shapeDataType);
        const auto axis    = ngraph::opset3::Constant::create(shapeDataType, {1}, {0});
        const auto indices = ngraph::opset3::Constant::create(shapeDataType, {1}, {inDataShapes.shape.size() - 1});
        const auto outShapeDescriptorDynamicNode = std::make_shared<ngraph::opset3::Concat>(
                ngraph::OutputVector{
                        ngraph::opset3::Constant::create(shapeDataType, {1}, {1}),
                        ngraph::opset3::Constant::create(shapeDataType, {1}, {-1}),
                        std::make_shared<ngraph::opset3::Gather>(shapeOf, indices, axis)},
                0);

        return std::make_shared<ngraph::opset3::Reshape>(
                inputSubgraph, outShapeDescriptorDynamicNode, true);
    }
};

TEST_P(DSR_ReshapeWithDynamicDescriptor, CompareWithReference) {
    Run();
}

const std::vector<ReshapeTestParams> reshapeTestParams = {
        std::make_tuple(DataShapeWithUpperBound{{1, 750}, {1, 1000}}, true, ShapeDescriptor{-1, 1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 1}, {1000, 1}}, true, ShapeDescriptor{-1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 1}, {750, 1}}, true, ShapeDescriptor{-1, 1, 1, 1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 4}, {1000, 4}}, true, ShapeDescriptor{1, -1, 4}),
        std::make_tuple(DataShapeWithUpperBound{{750}, {1000}}, true, ShapeDescriptor{1, 1, -1}),
        std::make_tuple(DataShapeWithUpperBound{{800, 81, 4}, {1000, 81, 4}}, true, ShapeDescriptor{0, -1}),
        std::make_tuple(DataShapeWithUpperBound{{800, 256, 7, 7}, {1000, 256, 7, 7}}, true, ShapeDescriptor{0, -1}),
};

const std::vector<ngraph::element::Type> dataTypesVector = {
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReshape, DSR_ReshapeWithStaticDescriptor,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypesVector),
        ::testing::ValuesIn(reshapeTestParams),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

INSTANTIATE_TEST_SUITE_P(smoke_DynamicReshape, DSR_ReshapeWithDynamicDescriptor,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypesVector),
        ::testing::ValuesIn(reshapeTestParams),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
