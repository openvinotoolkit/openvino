// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

struct VariadicSplitTestCase {
    DataShapeWithUpperBound dataShapes;
    std::vector<int64_t> splitLengths;
    int64_t axis;
};

const auto combinations = testing::Combine(
    testing::Values(
            ngraph::element::f16),
    testing::Values(
            ngraph::element::i32),
    testing::Values(
            VariadicSplitTestCase{{{6, 12, 10}, {6, 12, 15}}, {1, 1, 3, 1}, 0},
            VariadicSplitTestCase{{{6, 12}, {10, 12}}, {7, 2, 1, 2}, 1},
            VariadicSplitTestCase{{{6, 12, 10, 24}, {6, 12, 10, 50}}, {4, 6}, 2},
            VariadicSplitTestCase{{{6, 12, 10, 24}, {6, 12, 10, 50}}, {4, 6}, -2}),
    testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
    DataType,
    DataType,
    VariadicSplitTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_VariadicSplit : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& variadicSplitSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, variadicSplitSetup.dataShapes);

        const auto axis = ngraph::opset3::Constant::create(idxType, {}, std::vector<int64_t>{variadicSplitSetup.axis});
        const auto splitLengths = ngraph::opset3::Constant::create(idxType,
                {variadicSplitSetup.splitLengths.size()}, std::vector<int64_t>{variadicSplitSetup.splitLengths});

        return std::make_shared<ngraph::opset3::VariadicSplit>(inputSubgraph, axis, splitLengths);
    }
};

TEST_P(DSR_VariadicSplit, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicGatherData, DSR_VariadicSplit, combinations);

}  // namespace
