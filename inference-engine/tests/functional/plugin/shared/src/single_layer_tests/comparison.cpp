// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "single_layer_tests/comparison.hpp"

using namespace LayerTestsDefinitions::ComparisonParams;

namespace LayerTestsDefinitions {
std::string ComparisonLayerTest::getTestCaseName(testing::TestParamInfo<ComparisonTestParams> obj) {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision inputsPrecision;
    ngraph::helpers::ComparisonTypes comparisonOpType;
    ngraph::helpers::InputLayerType secondInputType;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, inputsPrecision, comparisonOpType, secondInputType, netPrecision, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS0=" << CommonTestUtils::vec2str(inputShapes.first) << "_";
    results << "IS1=" << CommonTestUtils::vec2str(inputShapes.second) << "_";
    results << "inputsPRC=" << inputsPrecision.name() << "_";
    results << "comparisonOpType=" << comparisonOpType << "_";
    results << "secondInputType=" << secondInputType << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName;
    return results.str();
}

void ComparisonLayerTest::SetUp() {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision inputsPrecision;
    ngraph::helpers::ComparisonTypes comparisonOpType;
    ngraph::helpers::InputLayerType secondInputType;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, inputsPrecision, comparisonOpType, secondInputType, netPrecision, targetDevice, additional_config) = this->GetParam();

    auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputsPrecision);
    configuration.insert(additional_config.begin(), additional_config.end());

    auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first});

    auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
    if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
        inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondInput));
    }

    auto comparisonNode = ngraph::builder::makeComparison(inputs[0], secondInput, comparisonOpType);
    function = std::make_shared<ngraph::Function>(comparisonNode, inputs, "Comparison");
}


TEST_P(ComparisonLayerTest, ComparisonTests) {
    Run();
}
} // namespace LayerTestsDefinitions