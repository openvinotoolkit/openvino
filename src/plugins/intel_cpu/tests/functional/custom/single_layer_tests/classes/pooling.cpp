// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "pooling.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
std::string PoolingLayerCPUTest::getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
    ov::test::poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams, additionalConfig) = obj.param;

    utils::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

    std::ostringstream results;
    results << "IS=(";
    results << ov::test::utils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << ov::test::utils::vec2str(shape) << "_";
    }
    results << "Prc=" << inPrc << "_";
    switch (poolType) {
    case utils::PoolingTypes::MAX:
        results << "MaxPool_";
        break;
    case utils::PoolingTypes::AVG:
        results << "AvgPool_";
        results << "ExcludePad=" << excludePad << "_";
        break;
    }
    results << "K" << ov::test::utils::vec2str(kernel) << "_";
    results << "S" << ov::test::utils::vec2str(stride) << "_";
    results << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    results << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";
    results << "INT8=" << isInt8 << "_";
    if (!additionalConfig.empty()) {
        results << "_PluginConf";
            for (auto& item : additionalConfig) {
                results << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

    results << CPUTestsBase::getTestCaseName(cpuParams);
    results << CpuTestWithFusing::getTestCaseName(fusingParams);
    return results.str();
}

void PoolingLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    utils::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    if (isInt8)
        selectedType = selectedType + "_I8";
    else
        selectedType = makeSelectedTypeStr(selectedType, deduce_expected_precision(inPrc, configuration));

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }

    std::shared_ptr<ov::Node> poolInput = params[0];
    if (isInt8) {
        abs_threshold = 2e-2;
        ov::Shape newShape(poolInput->get_output_partial_shape(0).size(), 1);
        poolInput = ov::test::utils::make_fake_quantize(poolInput, inPrc, 256, newShape);
    }

    std::shared_ptr<ov::Node> pooling;
    if (ov::test::utils::PoolingTypes::MAX == poolType) {
        pooling = std::make_shared<ov::op::v1::MaxPool>(poolInput, stride, padBegin, padEnd, kernel, roundingType, padType);
    } else {
        pooling = std::make_shared<ov::op::v1::AvgPool>(poolInput, stride, padBegin, padEnd, kernel, excludePad, roundingType, padType);
    }

    function = makeNgraphFunction(inPrc, params, pooling, "PoolingCPU");
}

std::string AvgPoolingV14LayerCPUTest::getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
    ov::test::poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams, additionalConfig) = obj.param;

    utils::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

    std::ostringstream results;
    results << "IS=(";
    results << ov::test::utils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << ov::test::utils::vec2str(shape) << "_";
    }
    results << "Prc=" << inPrc << "_";
    results << "ExcludePad=" << excludePad << "_";
    results << "K" << ov::test::utils::vec2str(kernel) << "_";
    results << "S" << ov::test::utils::vec2str(stride) << "_";
    results << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    results << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";
    results << "INT8=" << isInt8 << "_";
    if (!additionalConfig.empty()) {
        results << "_PluginConf";
            for (auto& item : additionalConfig) {
                results << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

    results << CPUTestsBase::getTestCaseName(cpuParams);
    results << CpuTestWithFusing::getTestCaseName(fusingParams);
    return results.str();
}

void AvgPoolingV14LayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    utils::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    if (isInt8)
        selectedType = selectedType + "_I8";
    else
        selectedType = makeSelectedTypeStr(selectedType, deduce_expected_precision(inPrc, configuration));

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }

    std::shared_ptr<ov::Node> poolInput = params[0];
    if (isInt8) {
        abs_threshold = 2e-2;
        ov::Shape newShape(poolInput->get_output_partial_shape(0).size(), 1);
        poolInput = ov::test::utils::make_fake_quantize(poolInput, inPrc, 256, newShape);
    }

    auto pooling = std::make_shared<ov::op::v14::AvgPool>(poolInput, stride, padBegin, padEnd, kernel, excludePad, roundingType, padType);

    function = makeNgraphFunction(inPrc, params, pooling, "PoolingCPU");
}

std::string MaxPoolingV8LayerCPUTest::getTestCaseName(
    const testing::TestParamInfo<maxPoolV8LayerCpuTestParamsSet>& obj) {
    maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams, additionalConfig) = obj.param;

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    ov::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) =
        basicParamsSet;

    std::ostringstream results;
    results << "IS=(";
    results << ov::test::utils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << ov::test::utils::vec2str(shape) << "_";
    }
    results << "Prc=" << inPrc << "_";
    results << "MaxPool_";
    results << "K" << ov::test::utils::vec2str(kernel) << "_";
    results << "S" << ov::test::utils::vec2str(stride) << "_";
    results << "D" << ov::test::utils::vec2str(dilation) << "_";
    results << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    results << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";
    if (!additionalConfig.empty()) {
        results << "_PluginConf";
        for (auto& item : additionalConfig) {
            results << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    results << CPUTestsBase::getTestCaseName(cpuParams);
    return results.str();
}

void MaxPoolingV8LayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    ov::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) =
        basicParamsSet;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, deduce_expected_precision(inPrc, configuration));

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }
    auto pooling = std::make_shared<ov::op::v8::MaxPool>(params[0],
                                                         stride,
                                                         dilation,
                                                         padBegin,
                                                         padEnd,
                                                         kernel,
                                                         roundingType,
                                                         padType,
                                                         indexElementType,
                                                         axis);
    pooling->get_rt_info() = getCPUInfo();
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(pooling->output(0))};
    function = std::make_shared<ov::Model>(results, params, "MaxPooling");
}

std::string MaxPoolingV14LayerCPUTest::getTestCaseName(
const testing::TestParamInfo<maxPoolV8LayerCpuTestParamsSet>& obj) {
    maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams, additionalConfig) = obj.param;

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    ov::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) =
        basicParamsSet;

    std::ostringstream results;
    results << "IS=(";
    results << ov::test::utils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << ov::test::utils::vec2str(shape) << "_";
    }
    results << "Prc=" << inPrc << "_";
    results << "MaxPool_";
    results << "K" << ov::test::utils::vec2str(kernel) << "_";
    results << "S" << ov::test::utils::vec2str(stride) << "_";
    results << "D" << ov::test::utils::vec2str(dilation) << "_";
    results << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    results << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";
    if (!additionalConfig.empty()) {
        results << "_PluginConf";
        for (auto& item : additionalConfig) {
            results << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }

    results << CPUTestsBase::getTestCaseName(cpuParams);
    return results.str();
}

void MaxPoolingV14LayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    ov::AnyMap additionalConfig;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ov::op::PadType padType;
    ov::op::RoundingType roundingType;
    ov::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) =
        basicParamsSet;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, deduce_expected_precision(inPrc, configuration));

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }
    auto pooling = std::make_shared<ov::op::v14::MaxPool>(params[0],
                                                         stride,
                                                         dilation,
                                                         padBegin,
                                                         padEnd,
                                                         kernel,
                                                         roundingType,
                                                         padType,
                                                         indexElementType,
                                                         axis);
    pooling->get_rt_info() = getCPUInfo();
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(pooling->output(0))};
    function = std::make_shared<ov::Model>(results, params, "MaxPooling");
}

TEST_P(PoolingLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

TEST_P(AvgPoolingV14LayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

TEST_P(MaxPoolingV8LayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

TEST_P(MaxPoolingV14LayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

namespace Pooling {

// The combination of parameters: NCHW + CEIL gives an accuracy problem in ACL AvgPool
const ov::op::RoundingType expectedAvgRoundingType(const ov::op::RoundingType ceil_type) {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return ov::op::RoundingType::FLOOR;
#else
    return ceil_type;
#endif
}

const std::vector<poolSpecificParams>& paramsMax3D() {
    static const std::vector<poolSpecificParams> paramsMax3D = {
            poolSpecificParams{ utils::PoolingTypes::MAX, {2}, {2}, {0}, {0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {4}, {2}, {0}, {0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {2}, {1}, {0}, {0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {7}, {2}, {2}, {2},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
    };
    return paramsMax3D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV83D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV83D = {
            maxPoolV8SpecificParams{ {2}, {2}, {1}, {0}, {0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {7}, {2}, {1}, {2}, {2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT},
    };
    return paramsMaxV83D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV143D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV143D = {
            maxPoolV8SpecificParams{ {2}, {2}, {1}, {0}, {0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
            maxPoolV8SpecificParams{ {2}, {2}, {1}, {0}, {0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {7}, {2}, {1}, {2}, {2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT},
    };
    return paramsMaxV143D;
}

const std::vector<poolSpecificParams>& paramsAvg3D() {
    static const std::vector<poolSpecificParams> paramsAvg3D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
    };
    return paramsAvg3D;
}

const std::vector<poolSpecificParams>& paramsAvgV143D() {
    static const std::vector<poolSpecificParams> paramsAvgV143D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {3}, {2}, {0}, {0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {3}, {2}, {0}, {0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_UPPER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_LOWER, true },
    };
    return paramsAvgV143D;
}

const std::vector<ElementType>& inpOutPrecision() {
    static const std::vector<ElementType> inpOutPrecision = {ElementType::f32/*, ElementType::bf16*/};
    return inpOutPrecision;
}

const std::vector<poolSpecificParams>& paramsMax4D() {
    static const std::vector<poolSpecificParams> paramsMax4D = {
            poolSpecificParams{ utils::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::SAME_LOWER, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {11, 7}, {2, 2}, {2, 2}, {2, 2},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
    };
    return paramsMax4D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV84D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV84D = {
            maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {11, 7}, {2, 2}, {1, 1}, {2, 2}, {2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT},
    };
    return paramsMaxV84D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV144D = {
            maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
            maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {11, 7}, {2, 2}, {1, 1}, {2, 2}, {2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT},
    };
    return paramsMaxV144D;
}

const std::vector<InputShape>& inputShapes3D() {
    static const std::vector<InputShape> inputShapes3D = {
            { {}, {{3, 4, 64}} },
            { {}, {{2, 8, 12}} },
            { {}, {{1, 16, 12}} },
            { {}, {{1, 21, 4}} },
            { {}, {{1, 32, 8}} },
            {
                // dynamic
                {-1, -1, -1},
                // target
                {
                    {1, 32, 8},
                    {1, 21, 4},
                    {2, 8, 12}
                }
            },
            {
                // dynamic
                {{1, 5}, {4, 32}, {1, 64}},
                // target
                {
                    {3, 4, 64},
                    {1, 16, 12},
                    {1, 32, 8}
                }
            }
    };
    return inputShapes3D;
}

const std::vector<InputShape>& inputShapes4D() {
    static const std::vector<InputShape> inputShapes4D = {
            { {}, {{3, 4, 64, 64}} },
            { {}, {{2, 8, 8, 12}} },
            { {}, {{1, 16, 16, 12}} },
            { {}, {{1, 21, 8, 4}} },
            { {}, {{1, 32, 8, 8}} },
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {1, 32, 8, 8},
                    {1, 21, 8, 4},
                    {2, 8, 8, 12},
                    {1, 96, 125, 125}
                }
            },
            {
                // dynamic
                {{1, 5}, {4, 32}, {1, 64}, {1, 64}},
                // target
                {
                    {3, 4, 64, 64},
                    {1, 16, 16, 12},
                    {1, 32, 8, 8}
                }
            },
            {
                // dynamic
                {{1, 10}, 16, 8, 8},
                // target
                {
                    {1, 16, 8, 8},
                    {2, 16, 8, 8},
                }
            }
    };
    return inputShapes4D;
}

const std::vector<InputShape>& inputShapes5D() {
    static const std::vector<InputShape> inputShapes5D = {
            { {}, {{1, 4, 16, 16, 16}} },
            { {}, {{2, 8, 8, 8, 8}} },
            { {}, {{2, 16, 12, 16, 20}} },
            { {}, {{1, 19, 16, 20, 8}} },
            { {}, {{1, 32, 16, 8, 12}} },
            {
                // dynamic
                {-1, -1, -1, -1, -1},
                // target
                {
                    {2, 8, 8, 8, 8},
                    {1, 19, 16, 20, 8},
                    {1, 4, 16, 16, 16}
                }
            },
            {
                // dynamic
                {{1, 5}, {4, 32}, {1, 64}, {1, 64}, {1, 25}},
                // target
                {
                    {1, 4, 16, 16, 16},
                    {1, 32, 16, 8, 12},
                    {3, 16, 4, 8, 3}
                }
            }
    };
    return inputShapes5D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV85D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV85D = {
            maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {7, 11, 6}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT },
    };
    return paramsMaxV85D;
}

const std::vector<maxPoolV8SpecificParams>& paramsMaxV145D() {
    static const std::vector<maxPoolV8SpecificParams> paramsMaxV145DCeilTorch = {
            maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
            maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER },
            maxPoolV8SpecificParams{ {7, 11, 6}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT },
    };
    return paramsMaxV145DCeilTorch;
}

const std::vector<poolSpecificParams>& paramsAvg4D() {
    static const std::vector<poolSpecificParams> paramsAvg4D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_LOWER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_UPPER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_LOWER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
    };
    return paramsAvg4D;
}

const std::vector<poolSpecificParams>& paramsAvgV144D() {
    static const std::vector<poolSpecificParams> paramsAvgV144D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_LOWER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::EXPLICIT, true },
    };
    return paramsAvgV144D;
}

const std::vector<poolSpecificParams>& paramsAvg5D() {
    static const std::vector<poolSpecificParams> paramsAvg5D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_LOWER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_UPPER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_LOWER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4, 4, 4}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                                expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, true },
    };
    return paramsAvg5D;
}

const std::vector<poolSpecificParams>& paramsAvgV145D() {
    static const std::vector<poolSpecificParams> paramsAvgV145D = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_UPPER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::SAME_LOWER, true },
            poolSpecificParams{ utils::PoolingTypes::AVG, {4, 4, 4}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                                expectedAvgRoundingType(ov::op::RoundingType::CEIL_TORCH), ov::op::PadType::EXPLICIT, true },
    };
    return paramsAvgV145D;
}

const std::vector<poolSpecificParams>& paramsMax5D() {
    static const std::vector<poolSpecificParams> paramsMax5D = {
            poolSpecificParams{ utils::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::SAME_LOWER, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                ov::op::RoundingType::CEIL, ov::op::PadType::SAME_UPPER, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
            poolSpecificParams{ utils::PoolingTypes::MAX, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                                ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT, false },
    };
    return paramsMax5D;
}

const std::vector<poolSpecificParams>& paramsAvg4D_Large() {
    static const std::vector<poolSpecificParams> paramsAvg4D_Large = {
            poolSpecificParams{ utils::PoolingTypes::AVG, {65, 65}, {65, 65}, {0, 0}, {0, 0},
                                ov::op::RoundingType::FLOOR, ov::op::PadType::VALID, true },
    };
    return paramsAvg4D_Large;
}

const std::vector<InputShape>& inputShapes4D_Large() {
    static const std::vector<InputShape> inputShapes4D_Large = {
            {
                // dynamic
                {-1, -1, -1, -1},
                // target
                {
                    {1, 16, 65, 65},
                    {1, 8, 130, 130},
                    {1, 16, 65, 65}
                }
            },
    };
    return inputShapes4D_Large;
}

const CPUSpecificParams& expectedCpuConfigAnyLayout() {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    static const CPUSpecificParams acl = CPUSpecificParams{{}, {}, {"acl"}, "acl"};
    return acl;
#else
    static const CPUSpecificParams ref = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};
    return ref;
#endif
}

const std::vector<CPUSpecificParams>& vecCpuConfigsFusing_4D() {
    const auto avx2_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"};
    const auto avx512_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"};
    const auto acl_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"acl"}, "acl"};

    static const std::vector<CPUSpecificParams> vecCpuConfigsFusing_4D = {avx2_nhwc, avx512_nhwc, acl_nhwc, expectedCpuConfigAnyLayout()};
    return vecCpuConfigsFusing_4D;
}

}  // namespace Pooling
}  // namespace test
}  // namespace ov
