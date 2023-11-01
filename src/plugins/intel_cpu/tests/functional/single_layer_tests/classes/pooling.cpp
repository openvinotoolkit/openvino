// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "pooling.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string PoolingLayerCPUTest::getTestCaseName(const testing::TestParamInfo<poolLayerCpuTestParamsSet>& obj) {
    LayerTestsDefinitions::poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams) = obj.param;

    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
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
    case ngraph::helpers::PoolingTypes::MAX:
        results << "MaxPool_";
        break;
    case ngraph::helpers::PoolingTypes::AVG:
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

    results << CPUTestsBase::getTestCaseName(cpuParams);
    results << CpuTestWithFusing::getTestCaseName(fusingParams);
    return results.str();
}

void PoolingLayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    LayerTestsDefinitions::poolSpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    bool isInt8;
    CPUSpecificParams cpuParams;
    fusingSpecificParams fusingParams;
    std::tie(basicParamsSet, inputShapes, inPrc, isInt8, cpuParams, fusingParams) = this->GetParam();

    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
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
        selectedType = makeSelectedTypeStr(selectedType, inPrc);

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }

    std::shared_ptr<ngraph::Node> poolInput = params[0];
    if (isInt8) {
        ov::Shape newShape(poolInput->get_output_partial_shape(0).size(), 1);
        poolInput = ngraph::builder::makeFakeQuantize(poolInput, inPrc, 256, newShape);
    }

    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(poolInput,
                                                                         stride,
                                                                         padBegin,
                                                                         padEnd,
                                                                         kernel,
                                                                         roundingType,
                                                                         padType,
                                                                         excludePad,
                                                                         poolType);

    function = makeNgraphFunction(inPrc, params, pooling, "PoolingCPU");
}

std::string MaxPoolingV8LayerCPUTest::getTestCaseName(
    const testing::TestParamInfo<maxPoolV8LayerCpuTestParamsSet>& obj) {
    LayerTestsDefinitions::maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams) = obj.param;

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    ngraph::element::Type indexElementType;
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

    results << CPUTestsBase::getTestCaseName(cpuParams);
    return results.str();
}

void MaxPoolingV8LayerCPUTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    LayerTestsDefinitions::maxPoolV8SpecificParams basicParamsSet;
    InputShape inputShapes;
    ElementType inPrc;
    CPUSpecificParams cpuParams;
    std::tie(basicParamsSet, inputShapes, inPrc, cpuParams) = this->GetParam();

    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    ngraph::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) =
        basicParamsSet;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, inPrc);

    init_input_shapes({inputShapes});

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
    }
    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makeMaxPoolingV8(params[0],
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
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pooling->output(0))};
    function = std::make_shared<ngraph::Function>(results, params, "MaxPooling");
}

TEST_P(PoolingLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

TEST_P(MaxPoolingV8LayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pooling");
}

namespace Pooling {

// The combination of parameters: NCHW + CEIL gives an accuracy problem in ACL AvgPool
const ngraph::op::RoundingType expectedAvgRoundingType() {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return ngraph::op::RoundingType::FLOOR;
#else
    return ngraph::op::RoundingType::CEIL;
#endif
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax3D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax3D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {2}, {0}, {0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4}, {2}, {0}, {0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {1}, {0}, {0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
    };
    return paramsMax3D;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg3D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg3D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_UPPER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
    };
    return paramsAvg3D;
}

const std::vector<ElementType>& inpOutPrecision() {
    static const std::vector<ElementType> inpOutPrecision = {ElementType::f32/*, ElementType::bf16*/};
    return inpOutPrecision;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax4D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax4D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
    };
    return paramsMax4D;
}

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams>& paramsMaxV84D() {
    static const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV84D = {
            LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                            ngraph::element::Type_t::i32, 0,
                                                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER },
    };
    return paramsMaxV84D;
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

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams>& paramsMaxV85D() {
    static const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV85D = {
            LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                            ngraph::element::Type_t::i32, 0,
                                                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER },
    };
    return paramsMaxV85D;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg4D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_LOWER, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_UPPER, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_LOWER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_UPPER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
    };
    return paramsAvg4D;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg5D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_LOWER, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_UPPER, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_LOWER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::SAME_UPPER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4, 4}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                                expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, true },
    };
    return paramsAvg5D;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsMax5D() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax5D = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                                ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
    };
    return paramsMax5D;
}

const std::vector<LayerTestsDefinitions::poolSpecificParams>& paramsAvg4D_Large() {
    static const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_Large = {
            LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {65, 65}, {65, 65}, {0, 0}, {0, 0},
                                ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID, true },
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


}  // namespace Pooling
}  // namespace CPULayerTestsDefinitions