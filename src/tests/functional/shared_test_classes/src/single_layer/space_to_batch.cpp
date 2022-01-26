// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/space_to_batch.hpp"

namespace LayerTestsDefinitions {

std::string SpaceToBatchLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToBatchParamsTuple> &obj) {
    std::vector<size_t> inShapes;
    std::vector<int64_t> blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(blockShape, padsBegin, padsEnd, inShapes, netPrc, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inShapes) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "BS=" << CommonTestUtils::vec2str(blockShape) << "_";
    result << "PB=" << CommonTestUtils::vec2str(padsBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padsEnd) << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void SpaceToBatchLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int64_t> blockShape, padsBegin, padsEnd;
    InferenceEngine::Precision netPrecision;
    std::tie(blockShape, padsBegin, padsEnd, inputShape, netPrecision, inPrc, outPrc.front(), inLayout, outLayout, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto s2b = ngraph::builder::makeSpaceToBatch(paramOuts[0], ngPrc, blockShape, padsBegin, padsEnd);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2b)};
    function = std::make_shared<ngraph::Function>(results, params, "SpaceToBatch");
}
}  // namespace LayerTestsDefinitions
