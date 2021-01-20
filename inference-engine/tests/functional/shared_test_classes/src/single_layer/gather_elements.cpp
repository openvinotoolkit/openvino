// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/gather_elements.hpp"

namespace LayerTestsDefinitions {

std::string GatherElementsLayerTest::getTestCaseName(const testing::TestParamInfo<GatherElementsParams>& obj) {
    InferenceEngine::SizeVector dataShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int axis;
    int indices_axis_dim;
    std::string device;
    Config config;
    GatherElementsParamsSubset gatherElementsArgsSubset;
    std::tie(gatherElementsArgsSubset, dPrecision, iPrecision, device, config) = obj.param;
    std::tie(dataShape, axis, indices_axis_dim) = gatherElementsArgsSubset;

    std::ostringstream result;
    result << "DS=" << CommonTestUtils::vec2str(dataShape) << "_";
    result << "axis=" << axis << "_";
    result << "indices_axis_dim=" << indices_axis_dim << "_";
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

void GatherElementsLayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape;
    InferenceEngine::Precision dPrecision, iPrecision;

    int axis;
    int indices_axis_dim;
    GatherElementsParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, targetDevice, configuration) = this->GetParam();
    std::tie(dataShape, axis, indices_axis_dim) = gatherArgsSubset;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    auto params = ngraph::builder::makeParams(ngDPrc, {dataShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto dataNode = paramOuts[0];
    auto gatherElements = std::dynamic_pointer_cast<ngraph::opset6::GatherElements>(
                          ngraph::builder::makeGatherElements(dataNode, ngIPrc, axis, indices_axis_dim));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gatherElements)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherElements");
}

}  // namespace LayerTestsDefinitions
