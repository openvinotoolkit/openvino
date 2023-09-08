// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include "gtest/gtest.h"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
std::string TransposeLayerCPUTest::getTestCaseName(testing::TestParamInfo<TransposeLayerCPUTestParamSet> obj) {
    Precision netPrecision;
    InputShape inputShapes;
    std::vector<size_t> inputOrder;
    std::string targetDevice;
    CPUSpecificParams cpuParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS=(";
    for (const auto& shape : inputShapes.second) {
        result << ov::test::utils::vec2str(shape) << "_";
    }
    result << ")_";
    result << "inputOrder=" << ov::test::utils::vec2str(inputOrder) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second;
        }
    }
    result << CPUTestsBase::getTestCaseName(cpuParams);
    return result.str();
}

void TransposeLayerCPUTest::SetUp() {
    Precision netPrecision;
    InputShape inputShapes;
    std::vector<size_t> inputOrder;
    CPUSpecificParams cpuParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, inputOrder, netPrecision, targetDevice, additionalConfig, cpuParams) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    outType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    updateSelectedType("unknown", inType, configuration);

    init_input_shapes({inputShapes});

    auto params = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);

    const auto inputOrderOp =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape({inputOrder.size()}), inputOrder);
    const auto transpose = std::make_shared<ov::op::v1::Transpose>(params, inputOrderOp);
    transpose->get_rt_info() = getCPUInfo();
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};

    function = std::make_shared<ngraph::Function>(results, ov::ParameterVector{params}, "TransposeLayerCPUTest");
    functionRefs = ngraph::clone_function(*function);
}

TEST_P(TransposeLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Transpose");
}

namespace Transpose {
const std::vector<InferenceEngine::Precision>& netPrecisionsPerChannels() {
    static const std::vector<InferenceEngine::Precision> netPrecisionsPerChannels = {Precision::I8, Precision::FP32};
    return netPrecisionsPerChannels;
}

const std::vector<InputShape>& dynamicInputShapes4DC16() {
    static const std::vector<InputShape> dynamicInputShapes4DC16 = {InputShape{// dynamic
                                                                    {-1, 16, -1, -1},
                                                                    // target
                                                                    {{2, 16, 21, 10}, {3, 16, 11, 12}, {2, 16, 21, 10}}}};
    return dynamicInputShapes4DC16;
}

const std::vector<InputShape>& dynamicInputShapes4DC32() {
    static const std::vector<InputShape> dynamicInputShapes4DC32 = {InputShape{// dynamic
                                                                    {-1, 32, -1, -1},
                                                                    // target
                                                                    {{4, 32, 16, 14}, {16, 32, 5, 16}, {4, 32, 16, 14}}}};
    return dynamicInputShapes4DC32;
}

const std::vector<InputShape>& dynamicInputShapes4D() {
    static const std::vector<InputShape> dynamicInputShapes4D = {
        InputShape{// dynamic
                {ov::Dimension(1, 20), ov::Dimension(10, 40), ov::Dimension(10, 40), ov::Dimension(10, 40)},
                // target
                {{1, 32, 21, 10}, {2, 25, 11, 12}, {4, 15, 16, 14}, {7, 10, 20, 16}, {1, 32, 21, 10}}},
        InputShape{// dynamic
                {-1, -1, -1, -1},
                // target
                {{1, 24, 21, 8}, {2, 16, 11, 6}, {1, 24, 21, 8}}}
    };
    return dynamicInputShapes4D;
}

const std::vector<std::vector<size_t>>& inputOrder4D() {
    static const std::vector<std::vector<size_t>> inputOrder4D = {
            std::vector<size_t>{0, 1, 2, 3},
            std::vector<size_t>{0, 2, 3, 1},
            std::vector<size_t>{0, 2, 1, 3},
            std::vector<size_t>{1, 0, 2, 3},
            std::vector<size_t>{},
    };
    return inputOrder4D;
}
}  // namespace Transpose
}  // namespace CPULayerTestsDefinitions
