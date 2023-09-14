// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

std::string MatMulLayerCPUTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj) {
    MatMulLayerTestParamsSet basicParamsSet;
    MatMulNodeType nodeType;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;

    std::tie(basicParamsSet, nodeType, fusingParams, cpuParams) = obj.param;

    ElementType netType;
    ElementType inType, outType;
    ShapeRelatedParams shapeRelatedParams;
    ngraph::helpers::InputLayerType secondaryInputType;
    TargetDevice targetDevice;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) =
        basicParamsSet;

    std::ostringstream result;
    result << (nodeType == MatMulNodeType::MatMul ? "MatMul_" : "FullyConnected_");
    result << "IS=";
    for (const auto& shape : shapeRelatedParams.inputShapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << "TS=";
    for (const auto& shape : shapeRelatedParams.inputShapes) {
        result << "(";
        if (!shape.second.empty()) {
        auto itr = shape.second.begin();
        do {
            result << ov::test::utils::vec2str(*itr);
        } while (++itr != shape.second.end() && result << "_");
        }
        result << ")_";
    }
    result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
    result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
    result << "secondaryInputType=" << secondaryInputType << "_";
    result << "netPRC=" << netType << "_";
    result << "inPRC=" << inType << "_";
    result << "outPRC=" << outType << "_";
    result << "trgDev=" << targetDevice;
    result << "config=(";
    for (const auto& configEntry : additionalConfig) {
        result << configEntry.first << ", " << configEntry.second << ":";
    }
    result << ")";
    result << CpuTestWithFusing::getTestCaseName(fusingParams);
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

template<typename T>
void MatMulLayerCPUTest::transpose(T& shape) {
    IE_ASSERT(shape.size() > 1);
    std::swap(*(shape.end() - 1), *(shape.end() - 2));
}

void MatMulLayerCPUTest::SetUp() {
    MatMulLayerTestParamsSet basicParamsSet;
    MatMulNodeType nodeType;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;

    std::tie(basicParamsSet, nodeType, fusingParams, cpuParams) = this->GetParam();
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    ShapeRelatedParams shapeRelatedParams;
    ElementType netType;
    helpers::InputLayerType secondaryInputType;
    std::map<std::string, std::string> additionalConfig;

    std::tie(shapeRelatedParams, netType, inType, outType, secondaryInputType, targetDevice, additionalConfig) = basicParamsSet;

    init_input_shapes(shapeRelatedParams.inputShapes);

    bool transpA = shapeRelatedParams.transpose.first;
    bool transpB = shapeRelatedParams.transpose.second;

    if (transpA) {
        transpose(inputDynamicShapes[0]);
        for (auto& shapes : targetStaticShapes) {
        transpose(shapes[0]);
        }
    }
    if (transpB) {
        transpose(inputDynamicShapes[1]);
        for (auto& shapes : targetStaticShapes) {
        transpose(shapes[1]);
        }
    }

    const auto& inShapeA = inputDynamicShapes[0];
    const auto& inShapeB = inputDynamicShapes[1];

    // see comment in MatMul::canFuse
    if (!(nodeType == MatMulNodeType::MatMul &&
          std::get<0>(fusingParams) && std::get<0>(fusingParams)->getFusedOpsNames().find("(PerChannel)") != std::string::npos &&
          std::max(inShapeA.size(), inShapeB.size()) > 2))
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
        inType = outType = netType = ElementType::bf16;
    else
        inType = outType = netType;

    cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";
    selectedType = makeSelectedTypeStr(selectedType, outType);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netType, inShapeA)};

    auto matrixB = builder::makeDynamicInputLayer(netType, secondaryInputType, inShapeB);
    if (secondaryInputType == helpers::InputLayerType::PARAMETER) {
        params.push_back(std::dynamic_pointer_cast<opset1::Parameter>(matrixB));
    }
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));
    auto matMul = builder::makeMatMul(paramOuts[0], matrixB, transpA, transpB);
    function = makeNgraphFunction(netType, params, matMul, cpuNodeType);
    checkFusingPosition = false;
}

TEST_P(MatMulLayerCPUTest, CompareWithRefs) {
    // due to disabled BF16 fakequant fusing: src/plugins/intel_cpu/src/graph_optimizer.cpp#L755, skip this case
    if (inType == ElementType::bf16) {
    if (cpuNodeType == "FullyConnected") {
        if (priority[0].find("amx") != std::string::npos || priority[0] == "brgemm_avx512") {
        if (fusedOps.size() == 2 && fusedOps[0] == std::string("FakeQuantize") && fusedOps[1] == std::string("Relu")) {
            GTEST_SKIP() << "Skip MatMul BF16 FakeQuantization Fusing test" << std::endl;
        }
        }
    }
    }
    run();
    CheckPluginRelatedResults(compiledModel, cpuNodeType);
}

} // namespace CPULayerTestsDefinitions
