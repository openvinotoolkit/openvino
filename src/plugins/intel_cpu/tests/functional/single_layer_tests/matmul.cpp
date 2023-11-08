// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mat_mul.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ie_precision.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ov_models/builders.hpp"
#include <string>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

enum class MatMulNodeType {
    MatMul,
    FullyConnected
};

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,        // Network precision
        ElementType,        // Input precision
        ElementType,        // Output precision
        ngraph::helpers::InputLayerType,   // Secondary input type
        TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> MatMulLayerTestParamsSet;

using MatMulLayerCPUTestParamSet = std::tuple<MatMulLayerTestParamsSet,
                                              MatMulNodeType,
                                              fusingSpecificParams,
                                              CPUSpecificParams>;

class MatMulLayerCPUTest : public testing::WithParamInterface<MatMulLayerCPUTestParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulLayerCPUTestParamSet>& obj) {
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

protected:
     std::string cpuNodeType;

    template<typename T>
    void transpose(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
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
        auto matMul = builder::makeMatMul(params[0], matrixB, transpA, transpB);
        function = makeNgraphFunction(netType, params, matMul, cpuNodeType);
        checkFusingPosition = false;
    }
};

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

namespace {

/* ============= Common params ============= */
std::map<std::string, std::string> emptyAdditionalConfig;

std::vector<std::map<std::string, std::string>> additionalConfig {
#ifndef OV_CPU_WITH_MLAS
    // FP32 precision is covered by MLAS
    std::map<std::string, std::string>{/* empty config */},
#endif
    {{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}}
};

std::vector<std::map<std::string, std::string>> filterAdditionalConfig_Brgemm() {
#ifndef OV_CPU_WITH_MLAS
    // FP32 precision is covered by MLAS
    std::vector<std::map<std::string, std::string>> additionalConfig = {
        std::map<std::string, std::string>{/* empty config */}
    };
#else
    std::vector<std::map<std::string, std::string>> additionalConfig = {};
#endif
    if (with_cpu_x86_bfloat16()) {
        additionalConfig.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    }

    return additionalConfig;
}

std::vector<std::map<std::string, std::string>> filterAdditionalConfig_BrgemmAmx() {
    std::vector<std::map<std::string, std::string>> additionalConfig;
    if (with_cpu_x86_bfloat16()) {
        additionalConfig.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    }

    return additionalConfig;
}

const std::vector<ElementType> netPRCs {
    ElementType::f32,
    ElementType::bf16
};

std::vector<CPUSpecificParams> filterSpecificParams() {
    std::vector<CPUSpecificParams> specificParams;
    specificParams.push_back(CPUSpecificParams{{}, {}, {"jit_gemm"}, "jit_gemm"});

    return specificParams;
}

//For FP32 precision, FC has brgemm avx2 support but Matmul doen't have brgemm avx2.
//Need to specify tryBrgAVX2 based on test case.
std::vector<CPUSpecificParams> filterSpecificParams_Brgemm(bool tryBrgAVX2 = false) {
    std::vector<CPUSpecificParams> specificParams;
    if (with_cpu_x86_avx512_core()) {
        specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512"}, "brgemm_avx512"});
    } else if (tryBrgAVX2 && with_cpu_x86_avx2()) {
        specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx2"}, "brgemm_avx2"});
    }

    return specificParams;
}

std::vector<CPUSpecificParams> filterSpecificParams_BrgemmAmx() {
    std::vector<CPUSpecificParams> specificParams;
    if (with_cpu_x86_avx512_core_amx()) {
        specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx"});
    }

    return specificParams;
}


std::vector<CPUSpecificParams> filterSpecificParams_Brgconv1x1() {
    std::vector<CPUSpecificParams> specificParams;
    if (with_cpu_x86_avx512_core()) {
        specificParams.push_back(CPUSpecificParams{{}, {}, {/* brgconv_avx512_1x1 is not a part of fc impl list */}, "brgconv_avx512_1x1"});
    }

    return specificParams;
}

std::vector<CPUSpecificParams> filterSpecificParams_MLAS() {
    // replace with mlas primitive type
    std::vector<CPUSpecificParams> specificParams;
    specificParams.push_back(CPUSpecificParams{{}, {}, {"gemm_mlas"}, "gemm_mlas"});
    return specificParams;
}

/* ============= FullyConnected ============= */
namespace fullyConnected {

const std::vector<ShapeRelatedParams> IS2D_smoke = {
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

const std::vector<ShapeRelatedParams> IS2D_nightly = {
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

std::vector<fusingSpecificParams> fusingParamsSet2D_smoke {
// The following three patterns are convered by MLAS test
#ifndef OV_CPU_WITH_MLAS
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel,
#endif
        fusingFakeQuantizePerTensorRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet2D_Brgemm_smoke {
// The following three patterns are convered by MLAS test
#ifndef OV_CPU_WITH_MLAS
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel,
#endif
        fusingFakeQuantizePerTensorRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet2D_nightly {
        fusingRelu,
#ifndef OV_CPU_WITH_MLAS
        fusingScaleShift, //covered by MLAS
#endif
        fusingPReluPerTensor,
        fusingFakeQuantizePerChannelRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet2DBF16 {
        emptyFusingSpec,
        fusingBias,
        fusingRelu,
        fusingPReluPerTensor,
};

const auto testParams2D_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig)),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_smoke),
                                             ::testing::ValuesIn(filterSpecificParams()));

const auto testParams2DBF16_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke),
                                                                    ::testing::ValuesIn(netPRCs),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                    ::testing::ValuesIn(additionalConfig)),
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet2DBF16),
                                                 ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulLayerCPUTest, testParams2D_smoke, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_BF16, MatMulLayerCPUTest, testParams2DBF16_smoke, MatMulLayerCPUTest::getTestCaseName);

#ifdef OV_CPU_WITH_MLAS
std::vector<fusingSpecificParams> fusingParamsSet2D_MLAS_smoke {
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel
};

const auto testParams2D_MLAS_smoke = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_smoke),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig)),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_MLAS_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_MLAS, MatMulLayerCPUTest, testParams2D_MLAS_smoke, MatMulLayerCPUTest::getTestCaseName);
#endif

const auto testParams2D_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig)),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams()));

const auto testParams2DBF16_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly),
                                                                    ::testing::ValuesIn(netPRCs),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(ElementType::undefined),
                                                                    ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                    ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                    ::testing::ValuesIn(additionalConfig)),
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet2DBF16),
                                                 ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D, MatMulLayerCPUTest, testParams2D_nightly, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_BF16, MatMulLayerCPUTest, testParams2DBF16_nightly, MatMulLayerCPUTest::getTestCaseName);

#ifdef OV_CPU_WITH_MLAS
std::vector<fusingSpecificParams> fusingParamsSet2D_MLAS_nightly {
        fusingScaleShift
};
const auto testParams2D_MLAS_nightly = ::testing::Combine(::testing::Combine(::testing::ValuesIn(IS2D_nightly),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::Values(emptyAdditionalConfig)),
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_MLAS_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_MLAS, MatMulLayerCPUTest, testParams2D_MLAS_nightly, MatMulLayerCPUTest::getTestCaseName);
#endif

const std::vector<ShapeRelatedParams> IS3D_smoke = {
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

const std::vector<ShapeRelatedParams> IS3D_nightly = {
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{1, 32, 120}, {120, 50}}), {true, true}},

    {
        {
            {{-1, -1, -1}, {{1, 32, 120}, {1, 12, 120}}},
            {{120, 3}, {{120, 3}, {120, 3}}}
        },
        {false, false}
    },
    {
        {
            {{-1, -1, 50}, {{1, 2, 50}, {1, 10, 50}, {1, 2, 50}, {2, 2, 50}}},
            {{50, 7}, {{50, 7}, {50, 7}, {50, 7}, {50, 7}}}
        },
        {true, false}
    },
    {
        {
            {{-1, -1, 32}, {{1, 5, 32}, {1, 5, 32}}},
            {{32, 3}, {{32, 3}, {32, 3}}}
        },
        {false, true}
    },
};

std::vector<fusingSpecificParams> fusingParamsSet3D_smoke {
// The following three patterns are convered by MLAS test
#ifndef OV_CPU_WITH_MLAS
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel,
#endif
        fusingFakeQuantizePerChannel,
        fusingScaleShiftAndFakeQuantizePerChannel,
};

std::vector<fusingSpecificParams> fusingParamsSet3D_nightly {
        fusingFakeQuantizePerTensorRelu,
};

std::vector<fusingSpecificParams> fusingParamsSet3DBF16 {
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel,
};

const auto fullyConnectedParams3D_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::Values(emptyAdditionalConfig));

const auto fullyConnectedParams3DBF16_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_smoke),
                                                           ::testing::ValuesIn(netPRCs),
                                                           ::testing::Values(ElementType::undefined),
                                                           ::testing::Values(ElementType::undefined),
                                                           ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                           ::testing::ValuesIn(additionalConfig));

const auto testParams3D_smoke = ::testing::Combine(fullyConnectedParams3D_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_smoke),
                                             ::testing::ValuesIn(filterSpecificParams()));

const auto testParams3DBF16_smoke = ::testing::Combine(fullyConnectedParams3DBF16_smoke,
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet3DBF16),
                                                 ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulLayerCPUTest, testParams3D_smoke, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_BF16, MatMulLayerCPUTest, testParams3DBF16_smoke, MatMulLayerCPUTest::getTestCaseName);

#ifdef OV_CPU_WITH_MLAS
std::vector<fusingSpecificParams> fusingParamsSet3D_MLAS_smoke {
        emptyFusingSpec,
        fusingBias,
        fusingMultiplyPerChannel
};

const auto testParams3D_MLAS_smoke = ::testing::Combine(fullyConnectedParams3D_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_MLAS_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_MLAS()));
INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_MLAS, MatMulLayerCPUTest, testParams3D_MLAS_smoke, MatMulLayerCPUTest::getTestCaseName);
#endif

const auto fullyConnectedParams3D_nightly = ::testing::Combine(::testing::ValuesIn(IS3D_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::Values(emptyAdditionalConfig));

const auto fullyConnectedParams3DBF16_nightly = ::testing::Combine(::testing::ValuesIn(IS3D_nightly),
                                                           ::testing::ValuesIn(netPRCs),
                                                           ::testing::Values(ElementType::undefined),
                                                           ::testing::Values(ElementType::undefined),
                                                           ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                           ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                           ::testing::ValuesIn(additionalConfig));

const auto testParams3D_nightly = ::testing::Combine(fullyConnectedParams3D_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet3D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams()));

const auto testParams3DBF16_nightly = ::testing::Combine(fullyConnectedParams3DBF16_nightly,
                                                 ::testing::Values(MatMulNodeType::FullyConnected),
                                                 ::testing::ValuesIn(fusingParamsSet3DBF16),
                                                 ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_3D, MatMulLayerCPUTest, testParams3D_nightly, MatMulLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(nightly_FC_3D_BF16, MatMulLayerCPUTest, testParams3DBF16_nightly, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS2D_Brgemm_smoke = {
    // needed by 'IS2D_Brgconv1x1_smoke'
    {static_shapes_to_test_representation({{1, 120}, {120, 120}}), {true, false}},
    {static_shapes_to_test_representation({{1, 128}, {128, 166}}), {true, false}},

    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {true, false}},
    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {true, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

    {
        {
            {{-1, -1}, {{12, 16}, {25, 16}, {12, 16}, {25, 16}}},
            {{16, 35}, {{16, 35}, {16, 35}, {16, 35}, {16, 35}}}
        },
        {false, false}
    },
    {
        {
            {{{0, 50}, {0, 50}}, {{17, 48}, {15, 48}}},
            {{48, 15}, {{48, 15}, {48, 15}}}
        },
        {true, true}
    },
};

const std::vector<ShapeRelatedParams> IS2D_Brgemm_nightly = {
    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {false, false}},
    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {false, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {true, true}},

    {
        {
            {{-1, 128}, {{11, 128}, {20, 128}, {11, 128}, {15, 128}}},
            {{128, 11}, {{128, 11}, {128, 11}, {128, 11}, {128, 11}}}
        },
        {true, false}
    },
    {
        {
            {{{0, 50}, 32}, {{50, 32}, {23, 32}}},
            {{32, 21}, {{32, 21}, {32, 21}}}
        },
        {false, true}
    },
};

const auto fullyConnectedParams2D_Brgemm_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testParams2D_Brgemm_smoke = ::testing::Combine(fullyConnectedParams2D_Brgemm_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_Brgemm_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_Brgemm(true)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_Brgemm, MatMulLayerCPUTest, testParams2D_Brgemm_smoke, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS2D_Brgconv1x1_smoke = {
    {static_shapes_to_test_representation({{49, 120}, {120, 120}}), {true, false}},
    {static_shapes_to_test_representation({{79, 120}, {120, 120}}), {true, false}},

    {static_shapes_to_test_representation({{256, 188}, {188, 120}}), {true, false}},
    {static_shapes_to_test_representation({{256, 188}, {188, 120}}), {true, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 200}}), {false, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 200}}), {false, true}},

    {
        {
            // ip->brg->ip->brg
            // {1, 120} are covered in 'IS2D_Brgemm_smoke' which is ip
            // {49, 120}, {79, 120} are covered above which is brg1x1
            {{-1, -1}, {{1, 120}, {49, 120}, {1, 120}, {79, 120}}},
            {{120, 120}, {{120, 120}, {120, 120}, {120, 120}, {120, 120}}}
        },
        {false, false}
    },
    {
        {
            // ip->brg->ip(cached)->brg(cached)
            {{{0, 200}, {0, 200}}, {{1, 128}, {199, 128}, {1, 128}, {199, 128}}},
            {{128, 166}, {{128, 166}, {128, 166}}}
        },
        {true, true}
    },
};

const auto fullyConnectedParams2D_Brgconv1x1_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_Brgconv1x1_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::Values(emptyAdditionalConfig));

const auto testParams2D_Brgconv1x1_smoke = ::testing::Combine(fullyConnectedParams2D_Brgconv1x1_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_Brgemm_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_Brgconv1x1()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_Brgconv1x1, MatMulLayerCPUTest, testParams2D_Brgconv1x1_smoke, MatMulLayerCPUTest::getTestCaseName);


const std::vector<ShapeRelatedParams> IS3D_Brgconv1x1_smoke = {
    {static_shapes_to_test_representation({{2, 49, 120}, {120, 120}}), {true, false}},
    {static_shapes_to_test_representation({{4, 79, 120}, {120, 120}}), {true, false}},

    {static_shapes_to_test_representation({{1, 256, 188}, {188, 120}}), {true, false}},
    {static_shapes_to_test_representation({{2, 256, 188}, {188, 120}}), {true, true}},

    {static_shapes_to_test_representation({{2, 71, 128}, {128, 200}}), {false, false}},
    {static_shapes_to_test_representation({{3, 71, 128}, {128, 200}}), {false, true}},

    {
        {
            // ip->brg->ip->brg
            // {1, 1, 120}, {3, 1, 120} are covered in 'IS3D_smoke' which is ip
            // {2, 49, 120}, {4, 79, 120} are covered above which is brg1x1
            {{-1, -1, -1}, {{1, 1, 120}, {2, 49, 120}, {3, 1, 120}, {4, 79, 120}}},
            {{120, 120}, {{120, 120}, {120, 120}, {120, 120}, {120, 120}}}
        },
        {false, false}
    },
    {
        {
            // weights: Acb32a->Acb64a->Acb32a(cached)->Acb64a(cached)
            {{-1, -1, -1}, {{1, 54, 96}, {8, 54 * 2, 96}, {1, 54, 96}, {8, 54 * 2, 96}}},
            {{96, 128}, {{96, 128}, {96, 128}, {96, 128}, {96, 128}}}
        },
        {false, false}
    },
    {
        {
            // ip->brg->ip(cached)->brg(cached)
            {{{0, 200}, {0, 200}, {0, 200}}, {{1, 18, 128}, {2, 199, 128}, {3, 18, 128}, {4, 199, 128}}},
            {{128, 166}, {{128, 166}, {128, 166}}}
        },
        {true, true}
    },
};

const auto fullyConnectedParams3D_Brgconv1x1_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_Brgconv1x1_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::Values(emptyAdditionalConfig));

const auto testParams3D_Brgconv1x1_smoke = ::testing::Combine(fullyConnectedParams3D_Brgconv1x1_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_Brgemm_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_Brgconv1x1()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_Brgconv1x1, MatMulLayerCPUTest, testParams3D_Brgconv1x1_smoke, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS2D_Brgemm_Amx_smoke = {
    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {true, false}},
    {static_shapes_to_test_representation({{59, 16}, {16, 120}}), {true, true}},

    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, false}},
    {static_shapes_to_test_representation({{71, 128}, {128, 20}}), {false, true}},

    {
        {
            {{-1, -1}, {{12, 16}, {25, 16}, {12, 16}, {25, 16}}},
            {{16, 35}, {{16, 35}, {16, 35}, {16, 35}, {16, 35}}}
        },
        {false, false}
    },
    {
        {
            {{{0, 50}, {0, 50}}, {{17, 48}, {15, 48}}},
            {{48, 15}, {{48, 15}, {48, 15}}}
        },
        {true, true}
    },
};

const auto fullyConnectedParams2D_Brgemm_Amx_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_Amx_smoke),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testParams2D_Brgemm_Amx_smoke = ::testing::Combine(fullyConnectedParams2D_Brgemm_Amx_smoke,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_Brgemm_smoke),
                                             ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_Brgemm_Amx, MatMulLayerCPUTest, testParams2D_Brgemm_Amx_smoke, MatMulLayerCPUTest::getTestCaseName);


const auto fullyConnectedParams2D_Brgemm_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testParams2D_Brgemm_nightly = ::testing::Combine(fullyConnectedParams2D_Brgemm_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_Brgemm(true)));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_Brgemm, MatMulLayerCPUTest, testParams2D_Brgemm_nightly, MatMulLayerCPUTest::getTestCaseName);

const auto fullyConnectedParams2D_Brgemm_Amx_nightly = ::testing::Combine(::testing::ValuesIn(IS2D_Brgemm_nightly),
                                                       ::testing::Values(ElementType::f32),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(ElementType::undefined),
                                                       ::testing::Values(helpers::InputLayerType::CONSTANT),
                                                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                       ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testParams2D_Brgemm_Amx_nightly = ::testing::Combine(fullyConnectedParams2D_Brgemm_Amx_nightly,
                                             ::testing::Values(MatMulNodeType::FullyConnected),
                                             ::testing::ValuesIn(fusingParamsSet2D_nightly),
                                             ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(nightly_FC_2D_Brgemm_Amx, MatMulLayerCPUTest, testParams2D_Brgemm_Amx_nightly, MatMulLayerCPUTest::getTestCaseName);

} // namespace fullyConnected


/* ============= MatMul ============= */
namespace matmul {

const std::vector<ShapeRelatedParams> IS = {
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
    {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
    {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
    {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}},
};

const std::vector<ShapeRelatedParams> IS_Dynamic = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{55, 12}, {33, 7}}}, // input 0
            {{-1, -1}, {{12, 55}, {7, 33}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, -1}, {{60, 5}, {30, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {false, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },
};

const std::vector<ShapeRelatedParams> IS_Dynamic_nightly = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{{5, 15}, {1, 12}, {4, 15}}, {{10, 10, 10}, {5, 5, 5}}}, // input 0
            {{{1, 13}, {3, 15}, {1, 10}}, {{10, 10, 10}, {5, 5, 5}}} // input 1
        },
        {true, true}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {2, 10}, {3, 15}, -1, 16 }, {{ 2, 12, 4, 16 }, { 3, 12, 2, 16 }}}, // input 0
            {{ 1, 1, -1, 4 }, {{ 1, 1, 16, 4 }, { 1, 1, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 1, -1, 16 }, {{ 1, 1, 4, 16 }, { 1, 1, 2, 16 }}}, // input 0
            {{ {2, 5}, {3, 15}, -1, 4 }, {{ 2, 12, 16, 4 }, { 2, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 16 }, {{ 4, 16 }, { 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {2, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, {1, 15}, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ -1, 4 }, {{ 16, 4 }, { 16, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ {1, 3}, {1, 9}, {1, 5}, {1, 10} }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ {1, 5}, {1, 7}, {1, 8}, {1, 5} }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ 1, 7, 4, -1 }, {{ 1, 7, 4, 5 }, { 1, 7, 4, 4 }}}, // input 0
            {{ 1, 7, -1, 4 }, {{ 1, 7, 5, 4 }, { 1, 7, 4, 4 }}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, true}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {true, false}
    },

    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{ -1, 12, -1, 16 }, {{ 1, 12, 4, 16 }, { 2, 12, 2, 16 }}}, // input 0
            {{ {1, 5}, 12, -1, 4 }, {{ 1, 12, 16, 4 }, { 1, 12, 16, 4 }}}  // input 1
        },
        {false, true}
    },
};

std::vector<fusingSpecificParams> matmulFusingParams {
        emptyFusingSpec,
        fusingElu,
        fusingSqrt,
        fusingPReluPerTensor,
        fusingMultiplyPerChannel,
        fusingAddPerTensor,
        fusingBias,
        fusingFakeQuantizePerChannel,
        /* @todo FQ unfolds into FQ + Convert + Substract + Multiply after LPT,
         * so Relu cannot be fused in this case. Should be analysed */
        // fusingFakeQuantizePerChannelRelu,
        fusingFakeQuantizePerTensorRelu,
        fusingScaleShiftAndFakeQuantizePerChannel,
};

const auto matMulParams = ::testing::Combine(::testing::ValuesIn(IS),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParams = ::testing::Combine(matMulParams,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::ValuesIn(matmulFusingParams),
                                           ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Static, MatMulLayerCPUTest, testParams, MatMulLayerCPUTest::getTestCaseName);


const auto matMulParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Dynamic),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamic = ::testing::Combine(matMulParamsDynamic,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsDynamic_nightly = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_nightly),
                                             ::testing::ValuesIn(netPRCs),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(ElementType::undefined),
                                             ::testing::Values(helpers::InputLayerType::PARAMETER),
                                             ::testing::Values(ov::test::utils::DEVICE_CPU),
                                             ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamic_nightly = ::testing::Combine(matMulParamsDynamic_nightly,
                                           ::testing::Values(MatMulNodeType::MatMul),
                                           ::testing::Values(emptyFusingSpec),
                                           ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Dynamic, MatMulLayerCPUTest, testParamsDynamic_nightly, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS_Dynamic_Fusing = {
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1}, {{16, 12}, {33, 7}, {16, 12}}}, // input 0
            {{-1, 33}, {{12, 33}, {7, 33}, {12, 33}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}}, // input 0
            {{-1, 5}, {{60, 5}, {30, 5}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}}, // input 0
            {{-1, -1, -1, 25}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}  // input 1
        },
        {false, false}
    },
    {
        { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
            {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}, {10, 10, 10}}}, // input 0
            {{-1, -1, 5}, {{10, 10, 5}, {5, 5, 5}, {10, 10, 5}}}  // input 1
        },
        {false, false}
    },
};

const auto matMulParamsDynamicFusing = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_Fusing),
                                                        ::testing::ValuesIn(netPRCs),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(ElementType::undefined),
                                                        ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                        ::testing::ValuesIn(additionalConfig));

const auto testParamsDynamicFusing = ::testing::Combine(matMulParamsDynamicFusing,
                                                  ::testing::Values(MatMulNodeType::MatMul),
                                                  ::testing::ValuesIn(matmulFusingParams),
                                                  ::testing::ValuesIn(filterSpecificParams()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Dynamic_Fusing, MatMulLayerCPUTest, testParamsDynamicFusing, MatMulLayerCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS_brgemm_smoke = {
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, false}},
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, false}},

        {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, true}},
        {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, true}},

        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},

        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}},
};

const std::vector<ShapeRelatedParams> IS_brgemm_nightly = {
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {false, true}},
        {static_shapes_to_test_representation({{1, 2, 32, 120}, {120, 5}}), {true, true}},

        {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {false, false}},
        {static_shapes_to_test_representation({{7, 32, 120}, {3, 7, 120, 50}}), {true, false}},

        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, true}},
        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, true}},

        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, false}},
        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, false}},
};

const auto matMulBrgemmParams_smoke = ::testing::Combine(::testing::ValuesIn(IS_brgemm_smoke),
                                                         ::testing::Values(ElementType::f32),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                         ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testBrgemmParams_smoke = ::testing::Combine(matMulBrgemmParams_smoke,
                                                       ::testing::Values(MatMulNodeType::MatMul),
                                                       ::testing::ValuesIn(matmulFusingParams),
                                                       ::testing::ValuesIn(filterSpecificParams_Brgemm()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Brgemm_Static, MatMulLayerCPUTest, testBrgemmParams_smoke, MatMulLayerCPUTest::getTestCaseName);

std::vector<fusingSpecificParams> matmulBrgemmAmxFusingParams {
        emptyFusingSpec,
        fusingPReluPerTensor,
        fusingAddPerTensor,
        fusingBias,
};

const std::vector<ShapeRelatedParams> IS_brgemm_Amx_smoke = {
        {static_shapes_to_test_representation({{1, 2, 32, 64}, {64, 5}}), {false, false}},
        {static_shapes_to_test_representation({{1, 2, 32, 64}, {64, 5}}), {true, false}},

        {static_shapes_to_test_representation({{7, 32, 128}, {3, 7, 128, 5}}), {false, true}},
        {static_shapes_to_test_representation({{7, 32, 128}, {3, 7, 128, 5}}), {true, true}},

        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {false, false}},
        {static_shapes_to_test_representation({{10, 10, 10}, {10, 10, 10}}), {true, false}},

        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {false, true}},
        {static_shapes_to_test_representation({{55, 12}, {12, 55}}), {true, true}},
};

const auto matMulBrgemmAmxParams_smoke = ::testing::Combine(::testing::ValuesIn(IS_brgemm_Amx_smoke),
                                                         ::testing::Values(ElementType::f32),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                         ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testBrgemmAmxParams_smoke = ::testing::Combine(matMulBrgemmAmxParams_smoke,
                                                       ::testing::Values(MatMulNodeType::MatMul),
                                                       ::testing::ValuesIn(matmulBrgemmAmxFusingParams),
                                                       ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Brgemm_Amx_Static, MatMulLayerCPUTest, testBrgemmAmxParams_smoke, MatMulLayerCPUTest::getTestCaseName);

const auto matMulBrgemmParams_nightly = ::testing::Combine(::testing::ValuesIn(IS_brgemm_nightly),
                                                         ::testing::Values(ElementType::f32),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                         ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testBrgemmParams_nightly = ::testing::Combine(matMulBrgemmParams_nightly,
                                                       ::testing::Values(MatMulNodeType::MatMul),
                                                       ::testing::ValuesIn(matmulFusingParams),
                                                       ::testing::ValuesIn(filterSpecificParams_Brgemm()));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Brgemm_Static, MatMulLayerCPUTest, testBrgemmParams_nightly, MatMulLayerCPUTest::getTestCaseName);

const auto matMulBrgemmAmxParams_nightly = ::testing::Combine(::testing::ValuesIn(IS_brgemm_Amx_smoke),
                                                         ::testing::Values(ElementType::f32),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(ElementType::undefined),
                                                         ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                         ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testBrgemmAmxParams_nightly = ::testing::Combine(matMulBrgemmAmxParams_nightly,
                                                       ::testing::Values(MatMulNodeType::MatMul),
                                                       ::testing::ValuesIn(matmulBrgemmAmxFusingParams),
                                                       ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(nightly_MM_Brgemm_Amx_Static, MatMulLayerCPUTest, testBrgemmAmxParams_nightly, MatMulLayerCPUTest::getTestCaseName);


const std::vector<ShapeRelatedParams> IS_Brgemm_Dynamic = {
        {
                {
                        {{-1, 256}, {{1, 256}}},
                        {{256, 384}, {{256, 384}}}
                },
                {false, false}
        },
        {
                {
                        {{-1, -1}, {{55, 12}, {33, 7}}},
                        {{-1, -1}, {{12, 55}, {7, 33}}}
                },
                {false, false}
        },
        {
                {
                        {{-1, -1, -1, -1}, {{1, 2, 32, 60}, {1, 2, 32, 30}}},
                        {{-1, -1}, {{60, 5}, {30, 5}}}
                },
                {true, false}
        },
        {
                {
                        {{-1, -1, -1}, {{7, 32, 60}, {7, 32, 30}}},
                        {{-1, -1, -1, -1}, {{3, 7, 60, 25}, {3, 7, 30, 25}}}
                },
                {false, true}
        },
        {
                {
                        {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}},
                        {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}
                },
                {false, false}
        },
        {
                {
                        {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}},
                        {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}}}
                },
                {true, true}
        },
        {
                {
                        {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}},
                        {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}
                },
                {true, false}
        },
        {
                {
                        {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}},
                        {{{1, 15}, {1, 15}, {1, 15}}, {{10, 10, 10}, {5, 5, 5}}}
                },
                {false, true}
        },
};

const auto matMulBrgemmParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Brgemm_Dynamic),
                                                          ::testing::Values(ElementType::f32),
                                                          ::testing::Values(ElementType::undefined),
                                                          ::testing::Values(ElementType::undefined),
                                                          ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                          ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testBrgemmParamsDynamic = ::testing::Combine(matMulBrgemmParamsDynamic,
                                                        ::testing::Values(MatMulNodeType::MatMul),
                                                        ::testing::Values(emptyFusingSpec),
                                                        ::testing::ValuesIn(filterSpecificParams_Brgemm()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Brgemm_Dynamic, MatMulLayerCPUTest, testBrgemmParamsDynamic, MatMulLayerCPUTest::getTestCaseName);

const auto matMulBrgemmAmxParamsDynamic = ::testing::Combine(::testing::ValuesIn(IS_Brgemm_Dynamic),
                                                          ::testing::Values(ElementType::f32),
                                                          ::testing::Values(ElementType::undefined),
                                                          ::testing::Values(ElementType::undefined),
                                                          ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                          ::testing::ValuesIn(filterAdditionalConfig_BrgemmAmx()));

const auto testBrgemmAmxParamsDynamic = ::testing::Combine(matMulBrgemmAmxParamsDynamic,
                                                        ::testing::Values(MatMulNodeType::MatMul),
                                                        ::testing::Values(emptyFusingSpec),
                                                        ::testing::ValuesIn(filterSpecificParams_BrgemmAmx()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Brgemm_Amx_Dynamic, MatMulLayerCPUTest, testBrgemmAmxParamsDynamic, MatMulLayerCPUTest::getTestCaseName);

const auto matMulParamsBrgemmDynamicFusing = ::testing::Combine(::testing::ValuesIn(IS_Dynamic_Fusing),
                                                                ::testing::Values(ElementType::f32),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(ElementType::undefined),
                                                                ::testing::Values(helpers::InputLayerType::PARAMETER),
                                                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                                                                ::testing::ValuesIn(filterAdditionalConfig_Brgemm()));

const auto testParamsBrgemmDynamicFusing = ::testing::Combine(matMulParamsBrgemmDynamicFusing,
                                                              ::testing::Values(MatMulNodeType::MatMul),
                                                              ::testing::ValuesIn(matmulFusingParams),
                                                              ::testing::ValuesIn(filterSpecificParams_Brgemm()));

INSTANTIATE_TEST_SUITE_P(smoke_MM_Brgemm_Dynamic_Fusing, MatMulLayerCPUTest, testParamsBrgemmDynamicFusing, MatMulLayerCPUTest::getTestCaseName);

} // namespace matmul

} // namespace

} // namespace CPULayerTestsDefinitions
