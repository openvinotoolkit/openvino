// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_activations.hpp"

namespace SubgraphTestsDefinitions {
std::string MultipleActivationsTest::getTestCaseName(const testing::TestParamInfo<MultipleActivationsParams> &obj) {
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

void MultipleActivationsTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(inputSize, netPrecision, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto input = ngraph::builder::makeParams(ngPrc, { {1, inputSize} });

    // auto relu = ngraph::builder::makeActivation(input[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);

    // auto concat_vals_1 = CommonTestUtils::generate_float_numbers(concatSize, 14, 14);
    // auto concat_vals_2 = CommonTestUtils::generate_float_numbers(concatSize, 14, 14);
    // auto concat_const_1 = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals_1);
    // auto concat_const_2 = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals_2);

    // auto concat_1 = ngraph::builder::makeConcat({concat_const_1, relu}, 1);
    // auto concat_2 = ngraph::builder::makeConcat({concat_const_2, relu}, 1);

    std::vector<size_t> outFormShapes = {1,  2 * inputSize};
    std::vector<float> fc1_weights = CommonTestUtils::generate_float_numbers(inputSize * outFormShapes[1], -0.5f, 0.5f);
    auto fc1 = ngraph::builder::makeFullyConnected(
        input[0], ngPrc, outFormShapes[1], false, {}, fc1_weights);

    auto tanh2 = ngraph::builder::makeActivation(fc1, ngPrc, ngraph::helpers::ActivationTypes::Tanh);

    auto eltw = ngraph::builder::makeEltwise(fc1, tanh2, ngraph::helpers::EltwiseTypes::ADD);

    function = std::make_shared<ngraph::Function>(eltw, input, "Muliple_Activations");
}
} // namespace SubgraphTestsDefinitions