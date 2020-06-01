// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <ie_core.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/subtract.hpp"

namespace LayerTestsDefinitions {
using namespace SubtractTestDefinitions;

const char* SubtractTestDefinitions::SecondaryInputType_to_string(SecondaryInputType input_type) {
    switch (input_type) {
    case SecondaryInputType::CONSTANT:
        return "CONSTANT";
    case SecondaryInputType::PARAMETER:
        return "PARAMETER";
    default:
        return "NOT_SUPPORTED_INPUT_LAYER_TYPE";
    }
}

const char* SubtractTestDefinitions::SubtractionType_to_string(SubtractionType subtraction_type) {
    switch (subtraction_type) {
    case SubtractionType::SCALAR:
        return "SCALAR";
    case SubtractionType::VECTOR:
        return "VECTOR";
    default:
        return "NOT_SUPPORTED_SUBSTRACTION_TYPE";
    }
}

std::string SubtractLayerTest::getTestCaseName(testing::TestParamInfo<SubtractParamsTuple> obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    SecondaryInputType secondaryInputType;
    SubtractionType SubtractionType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;

    std::tie(inputShapes, secondaryInputType, SubtractionType, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "secondaryInputType=" << SecondaryInputType_to_string(secondaryInputType) << "_";
    results << "subtractType=" << SubtractionType_to_string(SubtractionType) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void SubtractLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    SecondaryInputType secondaryInputType;
    SubtractionType subtractionType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, secondaryInputType, subtractionType, netPrecision, targetDevice, additional_config) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(ngPrc, {inputShapes[0]});

    std::vector<size_t> shape_input_secondary;
    switch (subtractionType) {
    case SubtractionType::SCALAR:
        shape_input_secondary = std::vector<size_t>({1});
        break;
    case SubtractionType::VECTOR:
        shape_input_secondary = inputShapes[0];
        break;
    default:
        FAIL() << "Unsupported SubtractionType: " << SubtractionType_to_string(subtractionType);
    }

    std::shared_ptr<ngraph::Node> secondary_input;
    switch (secondaryInputType) {
    case SecondaryInputType::CONSTANT:
        secondary_input = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, std::vector<float>{-1.0f});
        break;
    case SecondaryInputType::PARAMETER:
        input.push_back(ngraph::builder::makeParams(ngPrc, {shape_input_secondary})[0]);
        secondary_input = input[1];
        break;
    default:
        FAIL() << "Unsupported secondaryInputType: " << SecondaryInputType_to_string(secondaryInputType);
    }

    auto subtract = std::make_shared<ngraph::opset1::Subtract>(input[0], secondary_input);
    function = std::make_shared<ngraph::Function>(subtract, input, "subtraction");
}

TEST_P(SubtractLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
