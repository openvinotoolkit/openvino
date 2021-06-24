// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reshape.hpp"

namespace LayerTestsDefinitions {
std::string ReshapeLayerTest::getTestCaseName(testing::TestParamInfo<reshapeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::vector<int64_t> outFormShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    bool specialZero;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "OS=" << CommonTestUtils::vec2str(outFormShapes) << "_";
    result << "specialZero=" << specialZero << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ReshapeLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    std::vector<int64_t> outFormShapes;
    bool specialZero;
    InferenceEngine::Precision netPrecision;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, configuration) =
        this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto constNode = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::Type_t::i64, ngraph::Shape{outFormShapes.size()}, outFormShapes);
    auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
            std::make_shared<ngraph::opset1::Reshape>(paramIn[0], constNode, specialZero));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Reshape");
}

std::string ReshapeLayerTestRevise::getTestCaseName(testing::TestParamInfo<reshapeParamsRevise> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::vector<int64_t> outFormShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    bool specialZero;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "OS=" << CommonTestUtils::vec2str(outFormShapes) << "_";
    result << "specialZero=" << specialZero << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ReshapeLayerTestRevise::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    std::vector<int64_t> outFormShapes;
    bool specialZero;
    InferenceEngine::Precision netPrecision;
    std::tie(specialZero, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outFormShapes, targetDevice, configuration) =
        this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto constNode = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::Type_t::i64, ngraph::Shape{outFormShapes.size()}, outFormShapes);
    auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
            std::make_shared<ngraph::opset1::Reshape>(paramIn[0], constNode, specialZero));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reshape)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Reshape");
}

}  // namespace LayerTestsDefinitions
