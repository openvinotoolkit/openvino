// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/gather_nd.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset8.hpp"

namespace LayerTestsDefinitions {

std::string GatherNDLayerTest::getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    std::string device;
    Config config;
    GatherNDParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, device, config) = obj.param;
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    std::ostringstream result;
    result << "DS=" << ov::test::utils::vec2str(dataShape) << "_";
    result << "IS=" << ov::test::utils::vec2str(indicesShape) << "_";
    result << "BD=" << batchDims << "_";
    result << "DP=" << dPrecision.name() << "_";
    result << "IP=" << iPrecision.name() << "_";
    result << "device=" << device;
    if (!config.empty()) {
        result << "_config=";
        for (const auto& cfg : config) {
            result << "{" << cfg.first << ": " << cfg.second << "}";
        }
    }

    return result.str();
}

void GatherNDLayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    GatherNDParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, targetDevice, configuration) = this->GetParam();
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngDPrc, ov::Shape(dataShape))};
    auto paramOuts = ov::helpers::convert2OutputVector(
            ov::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto dataNode = paramOuts[0];
    auto gather = std::dynamic_pointer_cast<ngraph::opset5::GatherND>(
            ov::builder::makeGatherND(dataNode, indicesShape, ngIPrc, batchDims));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherND");
}


std::string GatherND8LayerTest::getTestCaseName(const testing::TestParamInfo<GatherNDParams>& obj) {
    return GatherNDLayerTest::getTestCaseName(obj);
}

void GatherND8LayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    GatherNDParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, targetDevice, configuration) = this->GetParam();
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngDPrc, ov::Shape(dataShape))};
    auto paramOuts = ov::helpers::convert2OutputVector(
        ov::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto dataNode = paramOuts[0];
    auto gather = std::dynamic_pointer_cast<ngraph::opset8::GatherND>(
        ov::builder::makeGatherND8(dataNode, indicesShape, ngIPrc, batchDims));
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(gather) };
    function = std::make_shared<ngraph::Function>(results, params, "gatherND");
}

}  // namespace LayerTestsDefinitions
