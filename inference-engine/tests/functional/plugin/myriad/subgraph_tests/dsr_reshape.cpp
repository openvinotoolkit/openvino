// Copyright (C) 2020 Intel Corporation
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


class DSR_Reshape : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(GetParam());
        const auto& reshapeTestParams = std::get<1>(GetParam());
        targetDevice = std::get<2>(GetParam());

        const auto& inDataShapes = std::get<0>(reshapeTestParams);
        const auto& specialZero = std::get<1>(reshapeTestParams);
        const auto& outShapeDescriptor = std::get<2>(reshapeTestParams);

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto outShapeDescriptorConstNode = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{outShapeDescriptor.size()}, outShapeDescriptor);
        const auto reshape = std::make_shared<ngraph::opset3::Reshape>(
                inputSubgraph, outShapeDescriptorConstNode, specialZero);

        return reshape;
    }
};

TEST_P(DSR_Reshape, CompareWithReference) {
    Run();
}

std::vector<ReshapeTestParams> reshapeTestParams = {
        std::make_tuple(DataShapeWithUpperBound{{1, 750}, {1, 1000}}, true, ShapeDescriptor{-1, 1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 1}, {1000, 1}}, true, ShapeDescriptor{-1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 1}, {750, 1}}, true, ShapeDescriptor{-1, 1, 1, 1}),
        std::make_tuple(DataShapeWithUpperBound{{750, 4}, {1000, 4}}, true, ShapeDescriptor{1, -1, 4}),
        std::make_tuple(DataShapeWithUpperBound{{750}, {1000}}, true, ShapeDescriptor{1, 1, -1}),
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicReshape, DSR_Reshape,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16,
                          ngraph::element::f32,
                          ngraph::element::i32),
        ::testing::ValuesIn(reshapeTestParams),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
