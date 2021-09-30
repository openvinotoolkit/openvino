// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/softmax.hpp"

namespace LayerTestsDefinitions {

std::string SoftMaxLayerTest::getTestCaseName(const testing::TestParamInfo<softMaxLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<ngraph::PartialShape> inputShape;
    std::vector<std::vector<InferenceEngine::SizeVector>> targetShapes;
    size_t axis;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShapes, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "axis=" << axis << "_";
    result << "trgDev=" << targetDevice;

    return result.str();
}

void SoftMaxLayerTest::SetUp() {
    std::vector<ngraph::PartialShape> inputShape;
    std::vector<std::vector<InferenceEngine::SizeVector>> targetShapes;

    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShapes, axis, targetDevice, configuration) = GetParam();
    outLayout = inLayout;

    for (auto&& targetShape : targetShapes) {
        targetStaticShapes.emplace_back(
                std::vector<ngraph::Shape>{ngraph::Shape{targetShape.front()}, ngraph::Shape{targetShape.front()}});
    }

    inputDynamicShape.emplace_back(inputShape.empty() ? targetStaticShapes[0].front() : inputShape[0]);

    setTargetStaticShape(targetStaticShapes[0]);

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShape.front()});
    const auto paramOuts =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(softMax)};

    function = std::make_shared<ngraph::Function>(results, params, "softMax");
    functionRefs = ngraph::clone_function(*function);
    functionRefs->set_friendly_name("softMaxRefs");
}

void SoftMaxLayerTest::setTargetStaticShape(std::vector<ngraph::Shape>& desiredTargetStaticShape) {
    targetStaticShape = desiredTargetStaticShape;
}

}  // namespace LayerTestsDefinitions
