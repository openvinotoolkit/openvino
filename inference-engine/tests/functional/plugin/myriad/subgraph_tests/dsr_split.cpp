// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

struct SplitTestCase {
    DataShapeWithUpperBound dataShapes;
    int64_t axis, numSplits;
};

const auto combinations = testing::Combine(
        testing::Values(
                ngraph::element::f16),
        testing::Values(
                ngraph::element::i32),
        testing::Values(
                SplitTestCase{{{6, 12, 10}, {6, 12, 15}}, 1, 3},
                SplitTestCase{{{6, 12, 10}, {9, 12, 10}}, 1, 3},
                SplitTestCase{{{6, 12}, {10, 12}}, 1, 4},
                SplitTestCase{{{6, 12, 10, 24}, {6, 12, 10, 50}}, 0, 6},
                SplitTestCase{{{6, 12, 10, 24}, {6, 12, 10, 50}}, -3, 2},
                SplitTestCase{{{1, 128, 4}, {1, 256, 4}}, 2, 4}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using Parameters = std::tuple<
        DataType,
        DataType,
        SplitTestCase,
        LayerTestsUtils::TargetDevice
>;

class DSR_Split : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& idxType = std::get<1>(parameters);
        const auto& splitSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, splitSetup.dataShapes);

        const auto axis = ngraph::opset5::Constant::create(idxType, {}, {splitSetup.axis});

        return std::make_shared<ngraph::opset5::Split>(inputSubgraph, axis, splitSetup.numSplits);
    }
};

TEST_P(DSR_Split, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSplit, DSR_Split, combinations);

}  // namespace
