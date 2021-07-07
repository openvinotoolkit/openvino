// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

struct ScatterTestCase {
    ngraph::NodeTypeInfo scatterTypeInfo;
    DataShapeWithUpperBound dataShapes, indicesShape, updatesShape;
    int64_t axis;
};

using Parameters = std::tuple<
    DataType,
    DataType,
    ScatterTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Scatter : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& numericType = std::get<0>(parameters);
        const auto& integerType = std::get<1>(parameters);
        const auto& scatterSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(numericType, scatterSetup.dataShapes);
        const auto indicesSubgraph = createInputSubgraphWithDSR(integerType, scatterSetup.indicesShape);
        const auto updatesSubgraph = createInputSubgraphWithDSR(numericType, scatterSetup.updatesShape);

        const auto axis = std::make_shared<ngraph::opset3::Constant>(integerType, ngraph::Shape{1}, std::vector<int64_t>{scatterSetup.axis});

        return ngraph::helpers::getNodeSharedPtr(scatterSetup.scatterTypeInfo, {inputSubgraph, indicesSubgraph, updatesSubgraph, axis});
    }
};

TEST_P(DSR_Scatter, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicScatter, DSR_Scatter,
    ::testing::Combine(
            testing::Values(
                    ngraph::element::f16),
            testing::Values(
                    ngraph::element::i32),
            testing::Values(
                    ScatterTestCase{
                        ngraph::opset3::ScatterUpdate::type_info,
                        {{84, 256, 7, 7}, {100, 256, 7, 7}},
                        {{84}, {100}},
                        {{84, 256, 7, 7}, {100, 256, 7, 7}},
                        0},
                    ScatterTestCase{
                        ngraph::opset5::ScatterElementsUpdate::type_info,
                        {{142}, {300}},
                        {{80}, {300}},
                        {{80}, {300}},
                        0}),
    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
