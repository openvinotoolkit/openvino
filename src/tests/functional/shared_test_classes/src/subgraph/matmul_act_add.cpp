// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_act_add.hpp"

namespace SubgraphTestsDefinitions {
std::string MatMulActAddTest::getTestCaseName(const testing::TestParamInfo<MatMulActAddParams> &obj) {
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

void MatMulActAddTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(inputSize, netPrecision, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> outFormShapes = {1,  2 * inputSize};

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{ 1, inputSize })};

    auto mul_const = ngraph::builder::makeConstant<float>(ngPrc, { outFormShapes[1], inputSize },
        ov::test::utils::generate_float_numbers(outFormShapes[1] * inputSize, -0.5f, 0.5f), false);

    auto matmul = std::make_shared<ngraph::op::MatMul>(params[0], mul_const, false, true);

    auto tanh = std::make_shared<ngraph::op::Tanh>(matmul);
    auto eltw = std::make_shared<ngraph::opset8::Add>(matmul, tanh);
    auto res = std::make_shared<ngraph::op::Result>(eltw);
    function = std::make_shared<ngraph::Function>(res, params, "MatMul_Act_Add");
}
} // namespace SubgraphTestsDefinitions
