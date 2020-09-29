// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "single_layer_tests/logical.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace LayerTestsDefinitions;

namespace {

typedef std::map<std::string, std::string> Config;

class LogicalLayerTestVPU : public LogicalLayerTest {
protected:
    void SetUp() override {
        const auto& inputShapes = std::get<0>(GetParam());
        const auto& ngInputsPrecision = std::get<1>(GetParam());
        const auto& logicalOpType = std::get<2>(GetParam());
        targetDevice = std::get<5>(GetParam());
        const auto& additionalConfig = std::get<6>(GetParam());
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        outPrc = ngInputsPrecision;

        auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(ngInputsPrecision);
        auto inputs = ngraph::builder::makeParams(ngInputsPrc, {inputShapes.first, logicalOpType != ngraph::helpers::LogicalTypes::LOGICAL_NOT ?
                                                                                   inputShapes.second : ngraph::Shape()});
        ngraph::NodeVector convertedInputs;
        for (const auto& input : inputs) {
            convertedInputs.push_back(std::make_shared<ngraph::opset5::Convert>(input, ngraph::element::boolean));
        }

        const auto logicalNode = ngraph::builder::makeLogical(convertedInputs[0], convertedInputs[1], logicalOpType);
        function = std::make_shared<ngraph::Function>(logicalNode, inputs, "Logical");
    }
};

TEST_P(LogicalLayerTestVPU, LogicalTests) {
    Run();
}

std::map<std::vector<size_t>, std::vector<std::vector<size_t >>> inputShapes = {
        {{5}, {{1}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t >>> inputShapesNot = {
        {{1}, {}},
        {{5}, {}},
        {{2, 200}, {}},
        {{1, 3, 20}, {}},
        {{2, 17, 3, 4}, {}},
        {{2, 1, 1, 3, 1}, {}},
};

std::vector<ngraph::helpers::LogicalTypes> eltwiseLogicalTypesInt = {
        ngraph::helpers::LogicalTypes::LOGICAL_AND,
};

INSTANTIATE_TEST_CASE_P(EltwiseLogicalInt,
                        LogicalLayerTestVPU,
                        ::testing::Combine(
                                ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapes)),
                                ::testing::Values(InferenceEngine::Precision::I32),
                                ::testing::ValuesIn(eltwiseLogicalTypesInt),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(EltwiseLogicalNotInt,
                        LogicalLayerTestVPU,
                        ::testing::Combine(
                                ::testing::ValuesIn(LogicalLayerTest::combineShapes(inputShapesNot)),
                                ::testing::Values(InferenceEngine::Precision::BOOL),
                                ::testing::Values(ngraph::helpers::LogicalTypes::LOGICAL_NOT),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        LogicalLayerTest::getTestCaseName);

} // namespace
