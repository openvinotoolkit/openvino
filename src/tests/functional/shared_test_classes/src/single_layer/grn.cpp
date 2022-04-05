// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/grn.hpp"

namespace LayerTestsDefinitions {
std::string GrnLayerTest::getTestCaseName(const testing::TestParamInfo<grnParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    float bias;
    std::tie(netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        bias,
        targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="     << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "bias="   << bias << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void GrnLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, bias, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { inputShapes });
    auto paramsOut = ngraph::helpers::convert2OutputVector(
        ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto grn = std::make_shared<ngraph::opset1::GRN>(paramsOut[0], bias);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(grn) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Grn");
}
}  // namespace LayerTestsDefinitions
