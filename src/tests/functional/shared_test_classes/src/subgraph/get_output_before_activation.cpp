// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/get_output_before_activation.hpp"

#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
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

std::string OutputBeforeActivationLegacy::getTestCaseName(const testing::TestParamInfo<OutputBeforeActivationLegacyParams>& obj) {
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

void OutputBeforeActivationLegacy::SetUp() {
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

InferenceEngine::Blob::Ptr OutputBeforeActivationLegacy::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, -1, 100);
}
} // namespace SubgraphTestsDefinitions

namespace ov {
namespace test {
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

std::string OutputBeforeActivation::getTestCaseName(const testing::TestParamInfo<OutputBeforeActivationLegacyParams> &obj) {
    std::string targetDevice;
    ov::element::Type type;
    size_t inputSize;
    midOutputType outputType;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, type, inputSize, outputType, config) = obj.param;
    std::ostringstream result;

    result << "IT=" << type.get_type_name() << "_";
    result << "IS=" << inputSize << "_";
    result << "OutputType=" << outputType << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void OutputBeforeActivation::SetUp() {
    std::map<std::string, std::string> config;
    size_t inputSize;
    midOutputType outputType;
    std::tie(targetDevice, inType, inputSize, outputType, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    std::vector<size_t> input_dims { 1, inputSize };

    auto input0 = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(input_dims));
    auto input1 = std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(input_dims));
    ov::ParameterVector params {input0, input1};

    ngraph::OutputVector outputs;
    std::shared_ptr<ngraph::Node> midLayer;
    switch (outputType) {
    case midOutputType::Sum: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::ADD);
        break;
    }
    case midOutputType::Sub: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::SUBTRACT);
        break;
    }
    case midOutputType::Mul: {
        midLayer = ngraph::builder::makeEltwise(input0, input1, ngraph::helpers::EltwiseTypes::MULTIPLY);
        break;
    }
    default:
        GTEST_FAIL() << "Unknown midOutputType";
    }

    auto act = ngraph::builder::makeActivation(midLayer, inType, ngraph::helpers::ActivationTypes::Tanh);
    outputs.insert(outputs.end(), {midLayer, act});
    function = std::make_shared<ov::Model>(outputs, params, "output_before_activation");

    std::vector<ov::test::InputShape> input_shapes;
    for (const auto& param : params) {
        input_shapes.push_back({{}, {param->get_shape()}});
    }
    init_input_shapes(input_shapes);
}

void OutputBeforeActivation::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto itTargetShape = targetInputStaticShapes.begin();
    for (const auto &param : function->get_parameters()) {
        auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape++, 2, -1, 100);
        inputs.insert({param, tensor});
    }
}
} //  namespace test
} //  namespace ov
