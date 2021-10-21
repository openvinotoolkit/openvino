// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/softmax.hpp"

namespace LayerTestsDefinitions {

std::string SoftMaxLayerTest::getTestCaseName(const testing::TestParamInfo<softMaxLayerTestParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>> shapes;
    size_t axis;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, axis, targetDevice, config) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
    result << "TS=";
    for (const auto& item : shapes.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }
    result << "axis=" << axis << "_";
    result << "trgDev=" << targetDevice;

    return result.str();
}

void SoftMaxLayerTest::SetUp() {
    std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>> shapes;
    InferenceEngine::Precision netPrecision;
    size_t axis;

    std::tie(netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, axis, targetDevice, configuration) = GetParam();
    outLayout = inLayout;

    targetStaticShapes.reserve(shapes.second.size());
    for (const auto& staticShape : shapes.second) {
        targetStaticShapes.push_back({staticShape});
    }
    inputDynamicShapes = {shapes.first};

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShapes.front().front()});
    const auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(softMax)};

    function = std::make_shared<ngraph::Function>(results, params, "softMax");
}
}  // namespace LayerTestsDefinitions