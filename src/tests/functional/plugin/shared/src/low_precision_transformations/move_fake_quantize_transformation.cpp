// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/move_fake_quantize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_lpt_models/move_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string MoveFakeQuantizeTransformation::getTestCaseName(testing::TestParamInfo<MoveFakeQuantizeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    std::vector<ngraph::PartialShape> inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    bool oneInputWithSplit;
    MoveFakeQuantizeTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, oneInputWithSplit, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShape[0], targetDevice, params) <<
        "SPLIT:" << oneInputWithSplit << "_" <<
        "OP:" << param.operation << "_" <<
        "FQ:" << param.fakeQuantizeAfter << "_" <<
        "DQ:" << param.dequantizationAfter;
    return result.str();
}

void MoveFakeQuantizeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    std::vector<ngraph::PartialShape> inputShapes;
    ov::pass::low_precision::LayerTransformation::Params params;
    bool oneInputWithSplit;
    MoveFakeQuantizeTransformationParam param;
    std::tie(netPrecision, inputShapes, targetDevice, params, oneInputWithSplit, param) = this->GetParam();

    function = ngraph::builder::subgraph::MoveFakeQuantize::get(
        netPrecision,
        inputShapes,
        param.concatInputsCount,
        {},
        {},
        {},
        param.operation,
        param.fakeQuantizeAfter,
        param.convertAfter,
        param.dequantizationAfter,
        {},
        {},
        param.axis,
        oneInputWithSplit);
}

void MoveFakeQuantizeTransformation::Run() {
    LayerTestsCommon::Run();

    const auto params = std::get<5>(GetParam());
    const auto actualPrecision = getRuntimePrecisionByType(params.layerName);
    auto expectedPrecision = params.expectedKernelType;
    if (expectedPrecision == "FP32" && std::get<0>(GetParam()) == ngraph::element::f16) {
        expectedPrecision = "FP16";
    }
    EXPECT_EQ(actualPrecision, expectedPrecision);
}

TEST_P(MoveFakeQuantizeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();
};

}  // namespace LayerTestsDefinitions
