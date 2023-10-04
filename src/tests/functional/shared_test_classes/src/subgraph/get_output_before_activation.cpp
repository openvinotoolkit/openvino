// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/get_output_before_activation.hpp"

namespace SubgraphTestsDefinitions {
std::ostream& operator<<(std::ostream& os, const midOutputType& oType) {
    switch (oType) {
    case midOutputType::Sub:
        return (os << "Sub");
    case midOutputType::Sum:
        return (os << "Sum");
    case midOutputType::Mul:
        return (os << "Mul");
    default:
        return (os << "Unknown");
    }
}

std::string OutputBeforeActivation::getTestCaseName(const testing::TestParamInfo<outputBeforeActivationParams>& obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    midOutputType outputType;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, outputType, config) = obj.param;
    std::ostringstream result;

    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "OutputType=" << outputType << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void OutputBeforeActivation::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    midOutputType outputType;
    std::tie(targetDevice, netPrecision, inputSize, outputType, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };

    ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims)),
                                         std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};
    auto input0 = input_parameter[0];
    auto input1 = input_parameter[1];

    ngraph::OutputVector outputs;
    std::shared_ptr<ngraph::Node> midLayer;
    switch (outputType) {
    case SubgraphTestsDefinitions::midOutputType::Sum: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::ADD);
        break;
    }
    case SubgraphTestsDefinitions::midOutputType::Sub: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::SUBTRACT);
        break;
    }
    case SubgraphTestsDefinitions::midOutputType::Mul: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::MULTIPLY);
        break;
    }
    default:
        GTEST_FAIL() << "Unknown midOutputType";
    }

    auto act = ngraph::builder::makeActivation(midLayer, ngPrc, ngraph::helpers::ActivationTypes::Tanh);
    outputs.insert(outputs.end(), {midLayer, act});
    function = std::make_shared<ngraph::Function>(outputs, input_parameter, "output_before_activation");
}

InferenceEngine::Blob::Ptr OutputBeforeActivation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, -1, 100);
}
} // namespace SubgraphTestsDefinitions
