// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/get_output_before_activation.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/test_enums.hpp"

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

std::string OutputBeforeActivation::getTestCaseName(const testing::TestParamInfo<outputBeforeActivationParams>& obj) {
    std::string targetDevice;
    ov::element::Type element_type;
    size_t inputSize;
    midOutputType outputType;
    ov::AnyMap config;
    std::tie(targetDevice, element_type, inputSize, outputType, config) = obj.param;
    std::ostringstream result;

    result << "InputType=" << element_type << "_";
    result << "IS=" << inputSize << "_";
    result << "OutputType=" << outputType << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second.as<std::string>();
    }
    return result.str();
}

void OutputBeforeActivation::SetUp() {
    ov::element::Type element_type;
    ov::AnyMap config;
    size_t inputSize;
    midOutputType outputType;
    std::tie(targetDevice, element_type, inputSize, outputType, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());

    std::vector<size_t> input_dims{1, inputSize};

    ov::ParameterVector input_parameter{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims)),
                                        std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(input_dims))};
    auto input0 = input_parameter[0];
    auto input1 = input_parameter[1];

    ov::OutputVector outputs;
    std::shared_ptr<ov::Node> midLayer;
    switch (outputType) {
    case ov::test::midOutputType::Sum: {
        midLayer = ov::test::utils::make_eltwise(input0, input1, ov::test::utils::EltwiseTypes::ADD);
        break;
    }
    case ov::test::midOutputType::Sub: {
        midLayer = ov::test::utils::make_eltwise(input0, input1, ov::test::utils::EltwiseTypes::SUBTRACT);
        break;
    }
    case ov::test::midOutputType::Mul: {
        midLayer = ov::test::utils::make_eltwise(input0, input1, ov::test::utils::EltwiseTypes::MULTIPLY);
        break;
    }
    default:
        GTEST_FAIL() << "Unknown midOutputType";
    }

    auto act = ov::test::utils::make_activation(midLayer, element_type, ov::test::utils::ActivationTypes::Tanh);
    outputs.insert(outputs.end(), {midLayer, act});
    function = std::make_shared<ov::Model>(outputs, input_parameter, "output_before_activation");
}

}  // namespace test
}  // namespace ov
