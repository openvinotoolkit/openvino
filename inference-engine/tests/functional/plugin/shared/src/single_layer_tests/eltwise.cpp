// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "single_layer_tests/eltwise.hpp"

namespace LayerTestsDefinitions {
const std::string EltwiseParams::InputLayerType_to_string(InputLayerType lt) {
    switch (lt) {
        case InputLayerType::CONSTANT:
            return "CONSTANT";
        case InputLayerType::PARAMETER:
            return "PARAMETER";
        default:
            return "NOT_SUPPORTED_INPUT_LAYER_TYPE";
    }
}

const std::string EltwiseParams::EltwiseOpType_to_string(EltwiseOpType eOp) {
    switch (eOp) {
        case EltwiseOpType::ADD:
            return "Sum";
        case EltwiseOpType::MULTIPLY:
            return "Prod";
        case EltwiseOpType::SUBSTRACT:
            return "Sub";
        default:
            return "NOT_SUPPORTED_ELTWISE_OPERATION";
    }
}

const std::string EltwiseParams::OpType_to_string(OpType op) {
    switch (op) {
        case OpType::SCALAR:
            return "SCALAR";
        case OpType::VECTOR:
            return "VECTOR";
        default:
            return "NOT_SUPPORTED_ADDITION_TYPE";
    }
}

std::string EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<EltwiseTestParams> obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    EltwiseParams::InputLayerType secondaryInputType;
    EltwiseParams::OpType opType;
    EltwiseParams::EltwiseOpType eltwiseOpType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseOpType, secondaryInputType, opType, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "eltwiseOpType=" << EltwiseParams::EltwiseOpType_to_string(eltwiseOpType) << "_";
    results << "secondaryInputType=" << EltwiseParams::InputLayerType_to_string(secondaryInputType) << "_";
    results << "opType=" << EltwiseParams::OpType_to_string(opType) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void EltwiseLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    EltwiseParams::InputLayerType secondaryInputType;
    EltwiseParams::OpType opType;
    EltwiseParams::EltwiseOpType eltwiseOpType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, eltwiseOpType, secondaryInputType, opType, netPrecision, targetDevice, additional_config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(ngPrc, {inputShapes[0]});

    std::vector<size_t> shape_input_secondary;
    switch (opType) {
        case EltwiseParams::OpType::SCALAR:
            shape_input_secondary = std::vector<size_t>({1});
            break;
        case EltwiseParams::OpType::VECTOR:
            shape_input_secondary = inputShapes[0];
            break;
        default:
            FAIL() << "Unsupported SubtractionType: " << EltwiseParams::OpType_to_string(opType);
    }

    std::shared_ptr<ngraph::Node> secondary_input;
    switch (secondaryInputType) {
        case EltwiseParams::InputLayerType::CONSTANT:
            secondary_input = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, std::vector<float>{-1.0f});
            break;
        case EltwiseParams::InputLayerType::PARAMETER:
            input.push_back(ngraph::builder::makeParams(ngPrc, {shape_input_secondary})[0]);
            secondary_input = input[1];
            break;
        default:
            FAIL() << "Unsupported secondaryInputType: " << EltwiseParams::InputLayerType_to_string(secondaryInputType);
    }

    switch (eltwiseOpType) {
        case EltwiseParams::EltwiseOpType::ADD: {
            auto add = std::make_shared<ngraph::opset1::Add>(input[0], secondary_input);
            function = std::make_shared<ngraph::Function>(add, input, "Add");
            break;
        }
        case EltwiseParams::EltwiseOpType::SUBSTRACT: {
            auto add = std::make_shared<ngraph::opset1::Subtract>(input[0], secondary_input);
            function = std::make_shared<ngraph::Function>(add, input, "Substract");
            break;
        }
        case EltwiseParams::EltwiseOpType::MULTIPLY: {
            auto add = std::make_shared<ngraph::opset1::Subtract>(input[0], secondary_input);
            function = std::make_shared<ngraph::Function>(add, input, "Multiply");
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "Incorrect type of Eltwise operation";
        }
    }
}


TEST_P(EltwiseLayerTest, EltwiseTests) {
    Run();
}
} // namespace LayerTestsDefinitions