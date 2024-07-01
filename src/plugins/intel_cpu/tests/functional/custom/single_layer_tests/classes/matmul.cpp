// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string MatMulLayerCPUTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj) {
    MatMulLayerTestParamsSet basicParamsSet;
    MatMulNodeType nodeType;
    fusingSpecificParams fusingParams;
    CPUSpecificParams cpuParams;

    std::tie(basicParamsSet, nodeType, fusingParams, cpuParams) = obj.param;

    ElementType netType;
    ElementType inType, outType;
    ShapeRelatedParams shapeRelatedParams;
    utils::InputLayerType secondaryInputType;
    TargetDevice targetDevice;
    ov::AnyMap additionalConfig;
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
        result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
    }
    result << ")";
    result << CpuTestWithFusing::getTestCaseName(fusingParams);
    result << CPUTestsBase::getTestCaseName(cpuParams);

    return result.str();
}

template<typename T>
void MatMulLayerCPUTest::transpose(T& shape) {
    OPENVINO_ASSERT(shape.size() > 1);
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
    utils::InputLayerType secondaryInputType;
    ov::AnyMap additionalConfig;

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

    auto it = additionalConfig.find(ov::hint::inference_precision.name());
    ov::element::Type inference_precision = (it != additionalConfig.end()) ?
                                            it->second.as<ov::element::Type>() : ov::element::undefined;
    if (inference_precision == ov::element::bf16) {
        inType = outType = netType = ElementType::bf16;
        rel_threshold = abs_threshold = 1e-2f;
    } else if (inference_precision == ov::element::f16) {
        inType = outType = netType = ElementType::f16;
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
        // rel_threshold = abs_threshold = 1e-2f;
        // Temporarily created the following rel_threshold because of this bug CVS-144523 and
        // https://github.com/ARM-software/ComputeLibrary/issues/1112
        rel_threshold = abs_threshold = 3e-1f;
#else
        rel_threshold = abs_threshold = 1e-4f;
#endif
    } else {
        inType = outType = netType;
        rel_threshold = 1e-4f;
        abs_threshold = 5e-4f;
    }

    cpuNodeType = nodeType == MatMulNodeType::MatMul ? "MatMul" : "FullyConnected";
    selectedType = makeSelectedTypeStr(selectedType, deduce_expected_precision(outType, configuration));

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netType, inShapeA)};

    std::shared_ptr<ov::Node> matrixB;
    if (secondaryInputType == utils::InputLayerType::PARAMETER) {
        auto param = std::make_shared<ov::op::v0::Parameter>(netType, inShapeB);
        matrixB = param;
        params.push_back(param);
    } else {
        ASSERT_TRUE(inShapeB.is_static());
        auto tensor = ov::test::utils::create_and_fill_tensor(netType, inShapeB.to_shape());
        matrixB = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    ov::OutputVector paramOuts;
    for (auto&& node : params) {
        for (auto&& param : node->outputs())
            paramOuts.push_back(param);
    }

    auto matMul = std::make_shared<ov::op::v0::MatMul>(paramOuts[0], matrixB, transpA, transpB);
    function = makeNgraphFunction(netType, params, matMul, cpuNodeType);
    checkFusingPosition = false;
}


void MatMulLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& parameters = function->get_parameters();
    for (size_t i = 0; i < parameters.size(); i++) {
        const auto& parameter = parameters[i];
        ov::Tensor tensor;
        const auto& param_type = parameter->get_output_element_type(0);
        const auto& static_shape = targetInputStaticShapes[i];
        switch (i) {
            case 0: {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = -4;
                    in_data.range = 8;
                    in_data.resolution = 32;
                    tensor = ov::test::utils::create_and_fill_tensor(param_type, static_shape, in_data);
                break;
            }
            case 1: {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 8;
                    in_data.resolution = 32;
                    tensor = ov::test::utils::create_and_fill_tensor(param_type, static_shape, in_data);
                break;
            }
            default: {
                throw std::runtime_error("Incorrect parameter number!");
            }
        }
        inputs.insert({parameter, tensor});
    }
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

namespace MatMul {
const ov::AnyMap& emptyAdditionalConfig() {
    static const ov::AnyMap emptyAdditionalConfig;
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

const std::vector<ov::AnyMap>& additionalConfig() {
    static std::vector<ov::AnyMap> additionalConfig{
#ifndef OV_CPU_WITH_MLAS
        // FP32 precision is covered by MLAS
        ov::AnyMap{/* empty config */},
#endif
        {ov::hint::inference_precision(ov::element::bf16)}};
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

}  // namespace MatMul
}  // namespace test
}  // namespace ov
