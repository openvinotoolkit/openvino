// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/concat.hpp"

namespace LayerTestsDefinitions {

std::string ConcatLayerTest::getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj) {
    int axis;
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(axis, inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "axis=" << axis << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConcatLayerTest::SetUp() {
    int axis;
    std::vector<std::vector<size_t>> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "concat");
}
}  // namespace LayerTestsDefinitions
