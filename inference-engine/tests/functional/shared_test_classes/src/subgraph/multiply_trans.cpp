// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiply_trans.hpp"

namespace SubgraphTestsDefinitions {
std::string MultiplyTransLayerTest::getTestCaseName(const testing::TestParamInfo<MultiplyTransParamsTuple> &obj) {
    std::vector<std::vector<size_t>> pairs;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(pairs, netPrecision, targetName) = obj.param;
    std::ostringstream results;
    const std::vector<size_t>& inputShapes = pairs[0];
    const std::vector<size_t>& transOrder = pairs[1];

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "TO=" << CommonTestUtils::vec2str(transOrder) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void MultiplyTransLayerTest::SetUp() {
    std::vector<std::vector<size_t>> pairs;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(pairs, netPrecision, targetDevice) = this->GetParam();
    const std::vector<size_t>& inputShape = pairs[0];
    const std::vector<size_t>& transOrder = pairs[1];

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto mulOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto const_mul = ngraph::builder::makeConstant<float>(ngPrc, inputShape, {}, true);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(mulOuts[0], const_mul);
    auto const_trans = ngraph::builder::makeConstant<size_t>(ngraph::element::i32, {transOrder.size()}, transOrder);
    auto trans = std::make_shared<ngraph::opset3::Transpose>(mul, const_trans);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(trans)};
    function = std::make_shared<ngraph::Function>(results, params, "multiplyTrans");
}
} // namespace SubgraphTestsDefinitions
