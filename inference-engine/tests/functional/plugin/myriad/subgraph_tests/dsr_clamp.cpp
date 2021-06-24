// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using Parameters = std::tuple<
    DataType,
    DataShapeWithUpperBound,
    LayerTestsUtils::TargetDevice
>;

class DSR_Clamp : public testing::WithParamInterface<Parameters>, public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& inDataType = std::get<0>(parameters);
        const auto& inDataShapes = std::get<1>(parameters);
        targetDevice = std::get<2>(parameters);

        const auto inputSubgraph = createInputSubgraphWithDSR(inDataType, inDataShapes);

        const auto clamp = std::make_shared<ngraph::opset3::Clamp>(inputSubgraph, 0., 6.);

        return clamp;
    }
};

TEST_P(DSR_Clamp, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicClamp, DSR_Clamp,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32),
        ::testing::Values(DataShapeWithUpperBound{DataShape{1, 800}, DataShape{2, 1000}}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
