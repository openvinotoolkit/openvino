// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/pad_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ov_lpt_models/pad.hpp"

namespace LayerTestsDefinitions {

std::string PadTransformation::getTestCaseName(const testing::TestParamInfo<PadTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::op::PadMode padMode;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    PadTransformationParam param;
    std::tie(netPrecision, inputShape, padMode, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape, targetDevice, params)
           << "_" << param.fakeQuantize << "_"
           << ov::test::utils::vec2str(param.padsBegin) << ov::test::utils::vec2str(param.padsEnd) << "_"
           << padMode << "_" << (padMode == ngraph::op::PadMode::CONSTANT ? "" : std::to_string(param.padValue));
    return result.str();
}

void PadTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShape;
    ngraph::op::PadMode mode;
    ov::pass::low_precision::LayerTransformation::Params params;
    PadTransformationParam param;
    std::tie(netPrecision, inputShape, mode, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::PadFunction::get(
        inputShape,
        netPrecision,
        param.fakeQuantize,
        param.padsBegin,
        param.padsEnd,
        mode,
        param.padValue);
}

void PadTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<5>(GetParam());
    const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
    const auto expectedPrecision = params.expectedKernelType;

    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(PadTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

} // namespace LayerTestsDefinitions
