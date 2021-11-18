// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

namespace {

using namespace LayerTestsUtils::vpu;

using AxisVector = std::vector<int64_t>;

struct UnsqueezeTestCase {
    DataShapeWithUpperBound inputShapes;
    AxisVector unsqueezeAxes;
};

using Parameters = std::tuple<
    DataType,
    UnsqueezeTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Unsqueeze : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& squeezeTestCase = std::get<1>(parameters);

        const auto& inputShapes = squeezeTestCase.inputShapes;
        const auto& unsqueezeAxes = squeezeTestCase.unsqueezeAxes;

        targetDevice = std::get<2>(GetParam());

        const auto inputSubgraph = createInputSubgraphWithDSR(dataType, inputShapes);
        const auto axes = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64, ngraph::Shape{unsqueezeAxes.size()}, unsqueezeAxes);

        return std::make_shared<ngraph::opset3::Unsqueeze>(inputSubgraph, axes);
    }
};

TEST_P(DSR_Unsqueeze, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicUnsqueeze, DSR_Unsqueeze,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::i32),
        ::testing::Values(
                // inputShapes, unsqueezeAxes
                UnsqueezeTestCase{DataShapeWithUpperBound{{789, 4}, {1000, 4}}, AxisVector{-1, -3}},
                UnsqueezeTestCase{DataShapeWithUpperBound{{789, 4}, {1000, 4}}, AxisVector{0}},
                UnsqueezeTestCase{DataShapeWithUpperBound{{789, 4}, {1000, 4}}, AxisVector{1}},
                UnsqueezeTestCase{DataShapeWithUpperBound{{789, 4}, {1000, 4}}, AxisVector{2}}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
