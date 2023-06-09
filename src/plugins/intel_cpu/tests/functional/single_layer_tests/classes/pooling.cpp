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
    results << CommonTestUtils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << CommonTestUtils::vec2str(shape) << "_";
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
    results << "K" << CommonTestUtils::vec2str(kernel) << "_";
    results << "S" << CommonTestUtils::vec2str(stride) << "_";
    results << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    results << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";
    results << "INT8=" << isInt8 << "_";

    results << CPUTestsBase::getTestCaseName(cpuParams);
    results << CpuTestWithFusing::getTestCaseName(fusingParams);
    return results.str();
}

void PoolingLayerCPUTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

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

    auto params = ngraph::builder::makeDynamicParams(inPrc, inputDynamicShapes);

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
    results << CommonTestUtils::partialShape2str({inputShapes.first}) << ")_";
    results << "TS=";
    for (const auto& shape : inputShapes.second) {
        results << CommonTestUtils::vec2str(shape) << "_";
    }
    results << "Prc=" << inPrc << "_";
    results << "MaxPool_";
    results << "K" << CommonTestUtils::vec2str(kernel) << "_";
    results << "S" << CommonTestUtils::vec2str(stride) << "_";
    results << "D" << CommonTestUtils::vec2str(dilation) << "_";
    results << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    results << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    results << "Rounding=" << roundingType << "_";
    results << "AutoPad=" << padType << "_";

    results << CPUTestsBase::getTestCaseName(cpuParams);
    return results.str();
}

void MaxPoolingV8LayerCPUTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;

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

    auto params = ngraph::builder::makeDynamicParams(inPrc, inputDynamicShapes);
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

namespace Pooling {}  // namespace Pooling
}  // namespace CPULayerTestsDefinitions