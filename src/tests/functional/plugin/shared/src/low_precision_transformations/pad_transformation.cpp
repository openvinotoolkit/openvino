// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/pad_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/pad.hpp"

namespace LayerTestsDefinitions {

std::string PadTransformation::getTestCaseName(const testing::TestParamInfo<PadTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::op::PadMode padMode;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    PadTransformationParam param;
    std::tie(netPrecision, inputShape, padMode, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params)
           << "_" << param.fakeQuantize << "_"
           << ov::test::utils::vec2str(param.padsBegin) << ov::test::utils::vec2str(param.padsEnd) << "_"
           << padMode << "_" << (padMode == ov::op::PadMode::CONSTANT ? "" : std::to_string(param.padValue));
    return result.str();
}

void PadTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::op::PadMode mode;
    ov::pass::low_precision::LayerTransformation::Params params;
    PadTransformationParam param;
    std::tie(netPrecision, inputShape, mode, targetDevice, params, param) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::PadFunction::get(
        inputShape,
        netPrecision,
        param.fakeQuantize,
        param.padsBegin,
        param.padsEnd,
        mode,
        param.padValue);
}

void PadTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<5>(GetParam());
    const auto actualPrecision = get_runtime_precision_by_type(params.layerName);
    const auto expectedPrecision = params.expectedKernelType;

    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(PadTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

} // namespace LayerTestsDefinitions
