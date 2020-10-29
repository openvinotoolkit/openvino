// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "single_layer_tests/gather_nd.hpp"

namespace LayerTestsDefinitions {

std::string GatherNDLayerTest::getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    std::string device;
    GatherNDParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, device) = obj.param;
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    std::ostringstream result;
    result << "DS=" << CommonTestUtils::vec2str(dataShape) << "_";
    result << "IS=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "BD=" << batchDims << "_";
    result << "DP=" << dPrecision.name() << "_";
    result << "IP=" << iPrecision.name() << "_";
    result << "device=" << device;
    return result.str();
}

void GatherNDLayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    GatherNDParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, targetDevice) = this->GetParam();
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    auto params = ngraph::builder::makeParams(ngDPrc, {dataShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto dataNode = paramOuts[0];
    auto gather = std::dynamic_pointer_cast<ngraph::opset5::GatherND>(
            ngraph::builder::makeGatherND(dataNode, indicesShape, ngIPrc, batchDims));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherND");
}

TEST_P(GatherNDLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
