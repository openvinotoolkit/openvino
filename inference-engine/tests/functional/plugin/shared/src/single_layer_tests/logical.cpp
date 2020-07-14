// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "single_layer_tests/logical.hpp"

using namespace LayerTestsDefinitions::LogicalParams;

namespace LayerTestsDefinitions {
std::string LogicalLayerTest::getTestCaseName(testing::TestParamInfo<LogicalTestParams> obj) {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision inputsPrecision;
    ngraph::helpers::LogicalTypes comparisonOpType;
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

std::vector<InputShapesTuple> LogicalLayerTest::combineShapes(const std::map<std::vector<size_t>, std::vector<std::vector<size_t >>>& inputShapes) {
    std::vector<InputShapesTuple> resVec;
    for (auto& inputShape : inputShapes) {
        for (auto& item : inputShape.second) {
            resVec.push_back({inputShape.first, item});
        }

        if (inputShape.second.empty()) {
            resVec.push_back({inputShape.first, {}});
        }
    }
    return resVec;
}


void LogicalLayerTest::SetUp() {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision inputsPrecision;
    ngraph::helpers::LogicalTypes logicalOpType;
    ngraph::helpers::InputLayerType secondInputType;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, inputsPrecision, logicalOpType, secondInputType, netPrecision, targetDevice, additional_config) = this->GetParam();

    auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputsPrecision);
    configuration.insert(additional_config.begin(), additional_config.end());

    auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first});

    std::shared_ptr<ngraph::Node> logicalNode;
    if (logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT) {
        auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
        if (secondInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            inputs.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondInput));
        }
        logicalNode = ngraph::builder::makeLogical(inputs[0], secondInput, logicalOpType);
    } else {
        logicalNode = ngraph::builder::makeLogical(inputs[0], ngraph::Output<ngraph::Node>(), logicalOpType);
    }

    function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
}


TEST_P(LogicalLayerTest, LogicalTests) {
    Run();
}
} // namespace LayerTestsDefinitions