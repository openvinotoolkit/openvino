// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/variadic_split_pad.hpp"

namespace SubgraphTestsDefinitions {

std::string VariadicSplitPad::getTestCaseName(const testing::TestParamInfo<SplitPadTuple> &obj) {
    InferenceEngine::SizeVector inputShape;
    size_t axis;
    std::vector<size_t> numSplits, connectIndexes;
    std::vector<int64_t> padsBegin, padsEnd;
    ngraph::helpers::PadMode padMode;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShape, axis, numSplits, connectIndexes, padsBegin, padsEnd, padMode, netPrecision, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    results << "Axis=" << axis << "_";
    results << "NumSplits=" << CommonTestUtils::vec2str(numSplits) << "_";
    results << "ConnectIndexes=" << CommonTestUtils::vec2str(connectIndexes) << "_";
    results << "padsBegin=" << CommonTestUtils::vec2str(padsBegin) << "_";
    results << "padsEnd=" << CommonTestUtils::vec2str(padsEnd) << "_";
    results << "PadMode=" << padMode << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void VariadicSplitPad::SetUp() {
    InferenceEngine::SizeVector inputs;
    size_t axis;
    std::vector<size_t> numSplits, connectIndexes;
    std::vector<int64_t> padBegin, padEnd;
    ngraph::helpers::PadMode padMode;
    InferenceEngine::Precision netPrecision;
    std::tie(inputs, axis, numSplits, connectIndexes, padBegin, padEnd, padMode, netPrecision, targetDevice) = this->GetParam();
    outPrc.front() = netPrecision;
    for (int i = 1; i< connectIndexes.size(); i++) {
        outPrc.push_back(netPrecision);
    }
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto input = ngraph::builder::makeParams(ngPrc, {inputs});
    auto split = ngraph::builder::makeVariadicSplit(input[0], numSplits, axis);
    ngraph::ResultVector results;

    for (size_t i : connectIndexes) {
        auto pad = ngraph::builder::makePad(split->output(i), padBegin, padEnd, 0, padMode);
        results.push_back(std::make_shared<ngraph::opset1::Result>(pad));
    }
    function = std::make_shared<ngraph::Function>(results, input, "variadic_split_pad");
}
} // namespace SubgraphTestsDefinitions
