// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <debug.h>
#include "ie_core.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "single_layer_tests/multiply.hpp"

namespace LayerTestsDefinitions {
using namespace MultiplyTestDefinitions;

std::string MultiplyLayerTest::getTestCaseName(const testing::TestParamInfo<MultiplyParamsTuple> &obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    SecondaryInputType secondaryInputType;
    MultiplicationType multiplicationType;
    std::string targetName;
    std::map<std::string, std::string> additional_config;

    std::tie(inputShapes, secondaryInputType, multiplicationType, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "secondaryInputType=" << SecondaryInputType_to_string(secondaryInputType) << "_";
    results << "mulType=" << MultiplicationType_to_string(multiplicationType) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void MultiplyLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    SecondaryInputType secondaryInputType;
    MultiplicationType multiplicationType;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, secondaryInputType, multiplicationType, netPrecision, targetDevice, additional_config) = this->GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());
    const std::size_t input_dim = InferenceEngine::details::product(inputShapes[0]);
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    std::vector<size_t> shape_input{1, input_dim};
    auto input = ngraph::builder::makeParams(ngPrc, {shape_input});

    std::vector<size_t> shape_input_secondary;
    switch(multiplicationType) {
    case MultiplicationType::SCALAR:
        shape_input_secondary = std::vector<size_t>({1});
        break;
    case MultiplicationType::VECTOR:
        shape_input_secondary = std::vector<size_t>(1, input_dim);
        break;
    default:
        FAIL() << "Unsupported MultiplicationType: " << MultiplicationType_to_string(multiplicationType); 
    }

    std::shared_ptr<ngraph::Node> secondary_input;
    switch(secondaryInputType) {
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

    auto mul = std::make_shared<ngraph::opset1::Multiply>(input[0], secondary_input);
    function = std::make_shared<ngraph::Function>(mul, input, "multiply");
}

TEST_P(MultiplyLayerTest, CompareWithRefs){
    Run();
};

const char* MultiplyTestDefinitions::SecondaryInputType_to_string(SecondaryInputType input_type) {
    switch (input_type) {
    case SecondaryInputType::CONSTANT:
        return "CONSTANT";
    case SecondaryInputType::PARAMETER:
        return "PARAMETER";
    default:
        return "NOT_SUPPORTED_INPUT_LAYER_TYPE";
    }
}

const char* MultiplyTestDefinitions::MultiplicationType_to_string(MultiplicationType multiplication_type) {
    switch (multiplication_type) {
    case MultiplicationType::SCALAR:
        return "SCALAR";
    case MultiplicationType::VECTOR:
        return "VECTOR";
    default:
        return "NOT_SUPPORTED_MULTIPLICATION_TYPE";
    }
}
} // namespace LayerTestsDefinitions
