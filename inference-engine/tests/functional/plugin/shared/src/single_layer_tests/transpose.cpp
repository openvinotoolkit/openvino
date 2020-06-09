// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_plugin_config.hpp>
#include <ie_core.hpp>
#include <functional>

#include "single_layer_tests/transpose.hpp"

namespace LayerTestsDefinitions {
    std::string TransposeLayerTest::getTestCaseName(testing::TestParamInfo<transposeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, inputOrder;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputShapes, inputOrder, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "inputOrder=" << CommonTestUtils::vec2str(inputOrder) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes, inputOrder;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShapes, inputOrder, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto constNode = std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::Type_t::i64, ngraph::Shape{inputOrder.size()}, inputOrder);
    auto transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
            std::make_shared<ngraph::opset1::Transpose>(paramIn[0], constNode));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(transpose)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Transpose");
}

TEST_P(TransposeLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
