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
    else if (additionalConfig[ov::hint::inference_precision.name()] == "f16")
        inType = outType = netType = ElementType::f16;
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
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

namespace MatMul {
const std::map<std::string, std::string>& emptyAdditionalConfig() {
    static const std::map<std::string, std::string> emptyAdditionalConfig;
    return emptyAdditionalConfig;
}

const std::vector<CPUSpecificParams>& filterSpecificParams() {
    static const std::vector<CPUSpecificParams> specificParams = {
        CPUSpecificParams{{}, {}, {"gemm_acl"}, "gemm_acl"},
        CPUSpecificParams{{}, {}, {"jit_gemm"}, "jit_gemm"}};
    return specificParams;
}

const std::vector<ElementType>& netPRCs() {
    static const std::vector<ElementType> netPRCs {
        ElementType::f32,
        ElementType::bf16
    };
    return netPRCs;
}

const std::vector<std::map<std::string, std::string>>& additionalConfig() {
    static std::vector<std::map<std::string, std::string>> additionalConfig {
    #ifndef OV_CPU_WITH_MLAS
        // FP32 precision is covered by MLAS
        std::map<std::string, std::string>{/* empty config */},
    #endif
        {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
    };
    return additionalConfig;
}

const std::vector<fusingSpecificParams>& matmulFusingParams() {
    static std::vector<fusingSpecificParams> matmulFusingParams {
            emptyFusingSpec,
            fusingElu,
            fusingSqrt,
            fusingPReluPerTensor,
            fusingMultiplyPerChannel,
    };
    return matmulFusingParams;
}

const std::vector<ShapeRelatedParams>& IS2D_nightly() {
    static const std::vector<ShapeRelatedParams> IS2D_nightly = {
        {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, false}},
        {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, false}},

        {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, false}},
        {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, true}},

        {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, true}},
        {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, true}},

        {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, true}},
        {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},

        {
            {
                {{-1, -1}, {{71, 128}, {50, 128}}},
                {{128, 20}, {{128, 20}, {128, 20}}}
            },
            {false, false}
        },
        {
            {
                {{-1, 59}, {{10, 59}, {15, 59}, {15, 59}}},
                {{59, 1}, {{59, 1}, {59, 1}, {59, 1}}}
            },
            {true, false}
        },
        {
            {
                {{{0, 120}, 59}, {{5, 59}, {11, 59}, {5, 59}, {10, 59}}},
                {{59, 120}, {{59, 120}, {59, 120}, {59, 120}, {59, 120}}}
            },
            {false, true}
        },
    };
    return IS2D_nightly;
}

const std::vector<ShapeRelatedParams>& IS2D_smoke() {
    static const std::vector<ShapeRelatedParams> IS2D_smoke = {
        {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {false, true}},
        {static_shapes_to_test_representation({{59, 1}, {1, 120}}), {true, true}},

        {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {false, false}},
        {static_shapes_to_test_representation({{59, 120}, {120, 1}}), {true, true}},

        {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {false, false}},
        {static_shapes_to_test_representation({{1, 120}, {120, 59}}), {true, false}},

        {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, false}},
        {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

        {
            {
                {{-1, -1}, {{20, 60}, {20, 60}}},
                {{60, 120}, {{60, 120}, {60, 120}}}
            },
            {false, false}
        },
        {
            {
                {{{0, 100}, {0, 12}}, {{20, 1}, {14, 1}, {20, 1}, {14, 1}}},
                {{1, 120}, {{1, 120}, {1, 120}, {1, 120}, {1, 120}}}
            },
            {true, true}
        },
    };
    return IS2D_smoke;
}

const std::vector<ShapeRelatedParams>& IS3D_smoke() {
    static const std::vector<ShapeRelatedParams> IS3D_smoke = {
        {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, false}},
        {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {false, true}},
        // needed by 'IS3D_Brgconv1x1_smoke'
        {static_shapes_to_test_representation({{1, 1, 120}, {120, 120}}), {false, false}},
        {static_shapes_to_test_representation({{3, 1, 120}, {120, 120}}), {false, false}},

        {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, false}},
        {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, true}},

        {
            {
                {{1, 5, 32}, {{1, 5, 32}, {1, 5, 32}}},
                {{32, 3}, {{32, 3}, {32, 3}}}
            },
            {false, true}
        },

        {static_shapes_to_test_representation({{1, 429}, {1, 429, 1}}), {true, true}},
        {
            {
                {{-1, -1}, {{1, 129}, {2, 129}, {1, 129}, {2, 129}}},
                {{1, 129, 1}, {{1, 129, 1}, {1, 129, 1}, {1, 129, 1}, {1, 129, 1}}}
            },
            {true, true}
        },

        {
            {
                {{{0, 60}, {0, 60}, {0, 60}}, {{1, 3, 14}, {1, 7, 14}}},
                {{14, 10}, {{14, 10}, {14, 10}}}
            },
            {true, true}
        },
    };
    return IS3D_smoke;
}

} // namespace MatMul
} // namespace CPULayerTestsDefinitions
