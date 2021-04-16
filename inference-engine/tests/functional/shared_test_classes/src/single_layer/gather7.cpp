// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather7.hpp"

namespace LayerTestsDefinitions {

void Gather7LayerTestBase::SetUp(const gather7ParamsTuple& params) {
    int64_t batch_dim;
    gatherParamsTuple gatherParams;
    std::tie(gatherParams, batch_dim) = params;
    int axis;
    std::vector<int> indices;
    std::vector<size_t> indicesShape;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = gatherParams;
    ASSERT_EQ(ngraph::shape_size(indicesShape), indices.size()) << "Indices vector size and provided indices shape doesn't fit each other";
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto functionParams = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(functionParams));
    auto indicesNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape(indicesShape), indices);
    auto axisNode = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape({}), {axis});
    auto gather = std::make_shared<ngraph::opset7::Gather>(paramOuts[0], indicesNode, axisNode, batch_dim);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, functionParams, "gather");
}

std::string Gather7LayerTest::getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple> &obj) {
    int64_t batch_dim;
    gatherParamsTuple gatherParams;
    std::tie(gatherParams, batch_dim) = obj.param;
    int axis;
    std::vector<int> indices;
    std::vector<size_t> indicesShape, inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::tie(indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = gatherParams;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "axis=" << axis << "_";
    result << "batch_dim=" << batch_dim << "_";
    result << "indices=" << CommonTestUtils::vec2str(indices) << "_";
    result << "indicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void Gather7LayerTest::SetUp() {
    Gather7LayerTestBase::SetUp(GetParam());
}

}  // namespace LayerTestsDefinitions
