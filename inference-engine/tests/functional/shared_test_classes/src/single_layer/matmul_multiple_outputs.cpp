// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/matmul_multiple_outputs.hpp"

namespace LayerTestsDefinitions {
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

    auto mul_const = ngraph::builder::makeConstant<float>(ngPrc, { outFormShapes[1], inputSize },
        CommonTestUtils::generate_float_numbers(outFormShapes[1] * inputSize, -0.5f, 0.5f), false);

    auto matmul = std::make_shared<ngraph::op::MatMul>(params[0], mul_const, false, true);

    auto tanh = ngraph::builder::makeActivation(matmul, ngPrc, ngraph::helpers::ActivationTypes::Tanh);

    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(matmul), std::make_shared<ngraph::op::Result>(tanh)};
    function = std::make_shared<ngraph::Function>(results, params, "MatMul_Multiple_Outputs");
}
} // namespace LayerTestsDefinitions
