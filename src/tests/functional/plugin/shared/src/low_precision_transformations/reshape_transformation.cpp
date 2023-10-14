// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape_transformation.hpp"

#include <memory>
#include <tuple>
#include <ie_core.hpp>

#include <transformations/init_node_info.hpp>
#include "ov_lpt_models/reshape.hpp"

namespace LayerTestsDefinitions {

std::string ReshapeTransformation::getTestCaseName(const testing::TestParamInfo<ReshapeTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << targetDevice << "_" << toString(params) <<
        "_" << param.inputShape << "_" << param.fakeQuantize << "_{";
    for (size_t i = 0; i < param.reshapeConstValues.size(); ++i) {
        result << param.reshapeConstValues[i];
        if (i != (param.reshapeConstValues.size() - 1ul)) {
            result << ", ";
        }
    }
    result << " }";
    return result.str();
}

void ReshapeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReshapeTransformationParam param;
    std::tie(netPrecision, targetDevice, params, param) = this->GetParam();

    function = ngraph::builder::subgraph::ReshapeFunction::getOriginal(
        param.inputShape,
        param.reshapeConstValues,
        netPrecision,
        param.fakeQuantize);
}

void ReshapeTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<3>(GetParam());
    auto actualPrecision = getRuntimePrecisionByType(params.layerType);
    const auto expectedPrecision = params.expectedKernelType;
    if ((expectedPrecision == "FP32") && (actualPrecision == "FP16")) {
        actualPrecision = "FP32";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(ReshapeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
