// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/einsum.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string EinsumLayerTest::getTestCaseName(const testing::TestParamInfo<EinsumLayerTestParamsSet>& obj) {
    InferenceEngine::Precision precision;
    EinsumEquationWithInput equationWithInput;
    std::string targetDevice;
    std::tie(precision, equationWithInput, targetDevice) = obj.param;
    std::string equation;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(equation, inputShapes) = equationWithInput;

    std::ostringstream result;
    result << "PRC=" << precision.name() << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Eq=" << equation << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void EinsumLayerTest::SetUp() {
    InferenceEngine::Precision precision;
    EinsumEquationWithInput equationWithInput;
    std::tie(precision, equationWithInput, targetDevice) = this->GetParam();
    std::string equation;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(equation, inputShapes) = equationWithInput;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(precision);
    const auto params = ngraph::builder::makeParams(ngPrc, inputShapes);
    const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const std::shared_ptr<ngraph::Node> einsum = ngraph::builder::makeEinsum(paramOuts, equation);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(einsum)};
    function = std::make_shared<ngraph::Function>(results, params, "einsum");
}

}  // namespace LayerTestsDefinitions
