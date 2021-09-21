// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_multiple_outputs.hpp"

namespace SubgraphTestsDefinitions {
std::string MatMulMultipleOutputsTest::getTestCaseName(const testing::TestParamInfo<MatMulMultipleOutputsParams> &obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    size_t inputSize;
    std::map<std::string, std::string> configuration;
    std::tie(inputSize, netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "inputSize=" << inputSize << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const &configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MatMulMultipleOutputsTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(inputSize, netPrecision, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, { {1, inputSize} });
    std::vector<size_t> outFormShapes = {1,  2 * inputSize};

    auto constant_1 = ngraph::builder::makeConstant<float>(ngPrc, { outFormShapes[1], inputSize },
        CommonTestUtils::generate_float_numbers(outFormShapes[1] * inputSize, -0.5f, 0.5f), false);

    auto matmul_1 = std::make_shared<ngraph::op::MatMul>(params[0], constant_1, false, true);

    auto tanh2 = ngraph::builder::makeActivation(matmul_1, ngPrc, ngraph::helpers::ActivationTypes::Tanh);

    auto eltw = ngraph::builder::makeEltwise(matmul_1, tanh2, ngraph::helpers::EltwiseTypes::ADD);

    function = std::make_shared<ngraph::Function>(eltw, params, "Muliple_Activations");
}
} // namespace SubgraphTestsDefinitions