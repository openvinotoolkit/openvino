// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#include "single_layer_tests/broadcast.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string BroadcastLayerTest::getTestCaseName(testing::TestParamInfo<broadcastLayerTestParamsSet> obj) {
    broadcastSpecificParams broadcastParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(broadcastParams, netPrecision, inputShapes, targetDevice) = obj.param;
    ngraph::op::BroadcastModeSpec mode;
    std::vector<size_t> targetShape, axesMapping;
    std::tie(mode, targetShape, axesMapping) = broadcastParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Mode=" << mode.m_type << "_";
    result << "TargetShape=" << CommonTestUtils::vec2str(targetShape) << "_";
    if (mode.m_type == ngraph::op::BroadcastType::EXPLICIT)
        result << "AxesMapping=" << CommonTestUtils::vec2str(axesMapping) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void BroadcastLayerTest::SetUp() {
    broadcastSpecificParams broadcastParams;
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(broadcastParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::BroadcastModeSpec mode;
    std::vector<size_t> targetShape, axesMapping;
    std::tie(mode, targetShape, axesMapping) = broadcastParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto broadcast = std::dynamic_pointer_cast<ngraph::opset3::Broadcast>(
            ngraph::builder::makeBroadcast(paramOuts[0], mode, targetShape, axesMapping));
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(broadcast)};
    function = std::make_shared<ngraph::Function>(results, params, "broadcast");
}

TEST_P(BroadcastLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
