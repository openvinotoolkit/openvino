// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/multi_input_scale.hpp"

namespace SubgraphTestsDefinitions {

std::string MultipleInputScaleTest::getTestCaseName(const testing::TestParamInfo<multipleInputScaleParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, config) = obj.param;
    std::ostringstream result;
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr MultipleInputScaleTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMin, range, 1 / inputDataResolution, seed);
}

void MultipleInputScaleTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(targetDevice, netPrecision, inputSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    std::vector<size_t> inputShape = {1, inputSize};
    ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape)),
                               std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto fc1_weights = ov::test::utils::generate_float_numbers(inputSize * inputSize, -0.5f, 0.5f);
    auto fc2_weights = ov::test::utils::generate_float_numbers(inputSize * inputSize, -0.2f, 0.2f);

    auto fc1 = ngraph::builder::makeFullyConnected(input[0], ngPrc, inputSize, false, {inputSize, inputSize}, fc1_weights);
    auto fc2 = ngraph::builder::makeFullyConnected(input[1], ngPrc, inputSize, false, {inputSize, inputSize}, fc2_weights);

    auto add = ngraph::builder::makeEltwise(fc1, fc2, ngraph::helpers::EltwiseTypes::ADD);

    auto result = std::make_shared<ngraph::opset7::Result>(add);
    function = std::make_shared<ngraph::Function>(result, input, "multiple_input_scale");
    functionRefs = ngraph::clone_function(*function);
}
}  // namespace SubgraphTestsDefinitions
