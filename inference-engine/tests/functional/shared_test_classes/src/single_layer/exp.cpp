// Copyright (C) 2021 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/exp.hpp"

namespace LayerTestsDefinitions {

std::string ExpLayerTest::getTestCaseName(testing::TestParamInfo<expParamsTuple> obj) {
    InferenceEngine::SizeVector inShape;
    InferenceEngine::Precision netPrc;
    std::string targetDevice;

    std::tie(inShape, netPrc, targetDevice) = obj.param;

    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(inShape) << "_";
    result << "netPrc=" << netPrc.name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void ExpLayerTest::SetUp() {
    InferenceEngine::SizeVector inShape;
    InferenceEngine::Precision netPrc;

    std::tie(inShape, netPrc, targetDevice) = this->GetParam();

    auto ngNetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrc);
    auto input = std::make_shared<ngraph::op::Parameter>(ngNetPrc, ngraph::Shape(inShape));
    auto exp = std::make_shared<ngraph::op::Exp>(input);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(exp), ngraph::ParameterVector{input});
}

} // namespace LayerTestsDefinitions
