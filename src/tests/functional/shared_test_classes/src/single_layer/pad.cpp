// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/pad.hpp"

namespace LayerTestsDefinitions {

std::string PadLayerTest::getTestCaseName(const testing::TestParamInfo<padLayerTestParamsSet>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout;
    InferenceEngine::SizeVector inputShapes;
    std::vector<int64_t> padsBegin, padsEnd;
    ngraph::helpers::PadMode padMode;
    float argPadValue;
    std::string targetDevice;
    std::tie(padsBegin, padsEnd, argPadValue, padMode, netPrecision, inPrc, outPrc, inLayout, inputShapes, targetDevice) =
      obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "padsBegin=" << ov::test::utils::vec2str(padsBegin) << "_";
    result << "padsEnd=" << ov::test::utils::vec2str(padsEnd) << "_";
    if (padMode == ngraph::helpers::PadMode::CONSTANT) {
        result << "Value=" << argPadValue << "_";
    }
    result << "PadMode=" << padMode << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void PadLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> padsBegin, padsEnd;
    float argPadValue;
    ngraph::helpers::PadMode padMode;
    InferenceEngine::Precision netPrecision;
    std::tie(padsBegin, padsEnd, argPadValue, padMode, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
    this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
    auto pad = CreatePadOp(paramOuts[0], padsBegin, padsEnd, argPadValue, padMode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pad)};
    function = std::make_shared<ngraph::Function>(results, params, "pad");
}
}  // namespace LayerTestsDefinitions
