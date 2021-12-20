// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using AxisVector = std::vector<int64_t>;

struct SqueezeTestCase {
    DataShapeWithUpperBound input_shape;
    AxisVector squeeze_axes;
};

using Parameters = std::tuple<
    DataType,
    SqueezeTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_Squeeze : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& squeezeTestCase = std::get<1>(parameters);

        const auto& inDataShapes = squeezeTestCase.input_shape;
        const auto& squeezeAxes = squeezeTestCase.squeeze_axes;

        targetDevice = std::get<2>(GetParam());

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto axes = std::make_shared<ngraph::opset3::Constant>(
                ngraph::element::i64, ngraph::Shape{squeezeAxes.size()}, squeezeAxes);
        return std::make_shared<ngraph::opset3::Squeeze>(inputSubgraph, axes);
    }
};

TEST_P(DSR_Squeeze, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSqueeze, DSR_Squeeze,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::i32),
        ::testing::Values(
                // input_shape, squeeze_axis
                SqueezeTestCase{DataShapeWithUpperBound{{1, 1, 1000}, {1, 1, 1500}}, AxisVector{-2}},
                SqueezeTestCase{DataShapeWithUpperBound{{1, 1000, 1}, {1, 1500, 1}}, AxisVector{0, 2}},
                SqueezeTestCase{DataShapeWithUpperBound{{1, 1, 1}, {2, 1, 2}}, AxisVector{1}},
                SqueezeTestCase{DataShapeWithUpperBound{{1000, 1, 1}, {1500, 1, 1}}, AxisVector{2}}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
