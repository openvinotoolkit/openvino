// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/* This test checks MatMul weights constant folding on CPU plugin side and cover two optimizations:
    1. Decompressing Convert FP16 -> FP32 CF (FuseFCAndConvertOnWeights in cpu graph optimizer)
    2. Transpose CF (FuseFCAndTransposeOnWeights in cpu graph optimizer)

 * 1. Graph with decompressing Convert FP16 -> FP32. The Convert node should be removed on the CPU plugin side.
 * Graph before:
   ------------             ------------
   |Input(f32)|             |Input(f16)|
   ------------             ------------
        |                        |
        |         ---------------------------------
        |         |Convert(decompression f16->f32)|
        |         ---------------------------------
        |                        |
    -----------------------------------------------
    |                   MatMul                    |
    -----------------------------------------------
                          |
                       --------
                       |Output|
                       --------

 * Exec graph:
   ------------    ------------
   |Input(f32)|    |Input(f16)|
   ------------    ------------
        |               |
   ----------------------------
   |      FullyConnected      |
   ----------------------------
                 |
              --------
              |Output|
              --------

 * 2. Graph with Transpose. In case of (transpose_b == false), ConvertMatMulToFC() transformation should insert Transpose on weights.
 * It must not fold and must remain in the execution graph.
 * Graph before:
   ------------             ------------
   |Input(f32)|             |Input(f32)|
   ------------             ------------
        |                        |
   -------------------------------------
   |   MatMul(transpose_b == false)    |
   -------------------------------------
                    |
                 --------
                 |Output|
                 --------

 * Exec graph:
   ------------    ------------
   |Input(f32)|    |Input(f32)|
   ------------    ------------
        |               |
        |         -------------
        |         | Transpose |
        |         -------------
        |               |
   ----------------------------
   |      FullyConnected      |
   ----------------------------
                 |
              --------
              |Output|
              --------
*/

using MatMulDecompressConvertParams = std::tuple<std::vector<InputShape>,  // input shapes
                                                 std::pair<bool, bool>,    // transposeA, transposeB
                                                 ElementType,              // weights precision
                                                 ov::AnyMap,               // additional config
                                                 CPUSpecificParams>;

class MatMulDecompressConvertTest : public testing::WithParamInterface<MatMulDecompressConvertParams>,
                                    virtual public SubgraphBaseTest,
                                    public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulDecompressConvertParams> obj) {
        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        ElementType weiElemType;
        ov::AnyMap additionalConfig;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, transpose, weiElemType, additionalConfig, cpuParams) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << transpose.first << "_";
        result << "transpose_b=" << transpose.second << "_";

        result << "weiLemType=" << weiElemType << "_";

        result << "config=(";
        for (const auto& configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second.as<std::string>() << ":";
        }
        result << ")";

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    template <typename T>
    void transpose_shape(T& shape) {
        OPENVINO_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void check_fc_weights_precision(ElementType expectedWeiElemType) const {
        auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string& paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        const auto execFunction = compiledModel.get_runtime_model();
        ASSERT_NE(nullptr, execFunction);
        for (const auto& fcNode : execFunction->get_ops()) {
            if (getExecValue(fcNode->get_rt_info(), ov::exec_model_info::LAYER_TYPE) == "FullyConnected") {
                const auto& constNode = fcNode->get_input_node_shared_ptr(1);
                ov::element::Type expectedType(
                    getExecValue(constNode->get_rt_info(), ov::exec_model_info::OUTPUT_PRECISIONS));
                ASSERT_EQ(expectedType, expectedWeiElemType);
            }
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        ElementType weiConstElemType;
        ov::AnyMap additionalConfig;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, transpose, weiConstElemType, additionalConfig, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        init_input_shapes(inputShapes);

        bool transpA = transpose.first;
        bool transpB = transpose.second;

        if (transpA)
            transposeCount++;
        if (!transpB)
            transposeCount++;

        if (transpA) {
            transpose_shape(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose_shape(shapes[0]);
            }
        }
        if (transpB) {
            transpose_shape(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose_shape(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ElementType netType = ElementType::f32;
        ElementType convertOutType = ElementType::f32;
        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
            convertOutType = inType = outType = netType = ElementType::bf16;
            weiConstElemType = (weiConstElemType != ElementType::f32) ? weiConstElemType : ElementType::bf16;
        } else {
            inType = outType = netType;
        }

        std::string cpuNodeType = "FullyConnected";
        selectedType = makeSelectedTypeStr(selectedType, outType);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inShapeA)};
        std::shared_ptr<ov::Node> inputB = ov::test::utils::make_constant(weiConstElemType, inShapeB.get_shape());
        if (weiConstElemType == ElementType::f16 || weiConstElemType == ElementType::bf16) {
            inputB = std::make_shared<ov::op::v0::Convert>(inputB, convertOutType);
            mark_as_decompression(inputB);
        }
        expectedWeiConstElemType = weiConstElemType;

        auto matMul = std::make_shared<ov::op::v0::MatMul>(params[0], inputB, transpA, transpB);

        function = CPUTestsBase::makeNgraphFunction(netType, params, matMul, cpuNodeType);
    }

    void check_execution_graph() {
        CheckPluginRelatedResults(compiledModel, "FullyConnected");
        CheckNumberOfNodesWithType(compiledModel, "FullyConnected", fullyConnectedCount);
        CheckNumberOfNodesWithType(compiledModel, "Transpose", transposeCount);
        CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
        CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
        check_fc_weights_precision(expectedWeiConstElemType);
    }

    size_t fullyConnectedCount = 1;
    size_t transposeCount = 0;
    ElementType expectedWeiConstElemType = ElementType::f32;
};

TEST_P(MatMulDecompressConvertTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_execution_graph();
}

namespace {

const std::vector<std::pair<bool, bool>> transposeParams = {
    {false, false},
    {false, true},
    {true, false},
    {true, true},
};

const std::vector<std::vector<InputShape>> inputShapes2D = {
    static_shapes_to_test_representation({{2, 3}, {3, 4}}),
    {{{-1, -1}, {{2, 3}, {5, 3}}}, {{3, 4}, {{3, 4}, {3, 4}}}},
};

const std::vector<std::vector<InputShape>> inputShapes3D = {
    static_shapes_to_test_representation({{2, 2, 3}, {3, 4}}),
    static_shapes_to_test_representation({{2, 3}, {1, 3, 4}}),
    static_shapes_to_test_representation({{1, 2, 3}, {1, 3, 4}}),
    {{{-1, -1, -1}, {{2, 2, 3}, {3, 5, 3}}}, {{3, 4}, {{3, 4}, {3, 4}}}},
    {{{-1, -1}, {{2, 3}, {5, 3}}}, {{1, 3, 4}, {{1, 3, 4}, {1, 3, 4}}}},
    {{{-1, -1, -1}, {{1, 2, 3}, {1, 5, 3}}}, {{1, 3, 4}, {{1, 3, 4}, {1, 3, 4}}}},
};

ov::AnyMap emptyConfig = {/* empty config */};

std::vector<ov::AnyMap> filter_additional_config_bf16() {
    std::vector<ov::AnyMap> additionalConfig;
    if (ov::with_cpu_x86_avx512_core()) {
        additionalConfig.push_back({{ov::hint::inference_precision(ov::element::bf16)}});
    }
    return additionalConfig;
}

std::vector<CPUSpecificParams> filter_specific_params(bool trySetMlas) {
    std::vector<CPUSpecificParams> specificParams;
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    specificParams.push_back(CPUSpecificParams{{}, {}, {"acl"}, "acl"});
#else
    if (trySetMlas) {
#ifdef OV_CPU_WITH_MLAS
        specificParams.push_back(CPUSpecificParams{{}, {}, {"gemm_mlas"}, "gemm_mlas"});
#endif
    }
    // try set onednn jit params if we can't or shouldn't use mlas
    if (specificParams.empty()) {
        if (ov::with_cpu_x86_avx512_core()) {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512"}, "brgemm_avx512"});
        } else if (ov::with_cpu_x86_avx2()) {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx2"}, "brgemm_avx2"});
        }
    }
#endif
    return specificParams;
}

std::vector<CPUSpecificParams> filter_specific_params_bf16() {
    std::vector<CPUSpecificParams> specificParams;
    specificParams.push_back(CPUSpecificParams{{}, {}, {"jit_gemm"}, "jit_gemm"});
    return specificParams;
}

const auto testParams2D_FP32_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes2D),
                                                        ::testing::ValuesIn(transposeParams),
                                                        ::testing::Values(ElementType::f32),
                                                        ::testing::Values(emptyConfig),
                                                        ::testing::ValuesIn(filter_specific_params(true)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_FP32,
                         MatMulDecompressConvertTest,
                         testParams2D_FP32_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams2D_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes2D),
                                                   ::testing::ValuesIn(transposeParams),
                                                   ::testing::Values(ElementType::f16, ElementType::bf16),
                                                   ::testing::Values(emptyConfig),
                                                   ::testing::ValuesIn(filter_specific_params(false)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D,
                         MatMulDecompressConvertTest,
                         testParams2D_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams2D_BF16_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes2D),
                                                        ::testing::ValuesIn(transposeParams),
                                                        ::testing::Values(ElementType::f32, ElementType::f16),
                                                        ::testing::ValuesIn(filter_additional_config_bf16()),
                                                        ::testing::ValuesIn(filter_specific_params_bf16()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_BF16,
                         MatMulDecompressConvertTest,
                         testParams2D_BF16_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams3D_FP32_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                                        ::testing::ValuesIn(transposeParams),
                                                        ::testing::Values(ElementType::f32),
                                                        ::testing::Values(emptyConfig),
                                                        ::testing::ValuesIn(filter_specific_params(true)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_FP32,
                         MatMulDecompressConvertTest,
                         testParams3D_FP32_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams3D_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                                        ::testing::ValuesIn(transposeParams),
                                                        ::testing::Values(ElementType::f16, ElementType::bf16),
                                                        ::testing::Values(emptyConfig),
                                                        ::testing::ValuesIn(filter_specific_params(false)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D,
                         MatMulDecompressConvertTest,
                         testParams3D_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams3D_BF16_smoke = ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                                        ::testing::ValuesIn(transposeParams),
                                                        ::testing::Values(ElementType::f32, ElementType::f16),
                                                        ::testing::ValuesIn(filter_additional_config_bf16()),
                                                        ::testing::ValuesIn(filter_specific_params_bf16()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_BF16,
                         MatMulDecompressConvertTest,
                         testParams3D_BF16_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

}  // namespace

/* This test covers decompression convert with several consumers.
 * Graph before:
   ------------              ------------            ------------
   |Input(f32)|              |Input(f16)|            |Input(f32)|
   ------------              ------------            ------------
        |                         |                         |
        |         ---------------------------------         |
        |         |Convert(decompression f16->f32)|         |
        |         ---------------------------------         |
        |             |                       |             |
    -----------------------               -----------------------
    |       MatMul        |               |       MatMul        |
    -----------------------               -----------------------
                      |                       |
                   ---------------------------------
                   |             Concat            |
                   ---------------------------------
                                   |
                                --------
                                |Output|
                                --------

 * Exec graph:
   ------------   --------------------------------   ------------
   |Input(f32)|   |           Input(f16)         |   |Input(f32)|
   ------------   --------------------------------   ------------
        |             |                       |             |
    -----------------------               -----------------------
    |       MatMul        |               |       MatMul        |
    -----------------------               -----------------------
                      |                       |
                   ---------------------------------
                   |             Concat            |
                   ---------------------------------
                                   |
                                --------
                                |Output|
                                --------
*/
using MatMulDecompressConvertParams2 = std::tuple<std::vector<InputShape>,  // input shapes
                                                  std::pair<bool, bool>,    // transposeA, transposeB
                                                  ElementType,              // weights precision
                                                  ov::AnyMap,               // additional property
                                                  CPUSpecificParams>;

class MatMulDecompressConvertTest2 : public MatMulDecompressConvertTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        ElementType weiConstElemType;
        ov::AnyMap additionalConfig;
        CPUSpecificParams cpuParams;

        std::tie(inputShapes, transpose, weiConstElemType, additionalConfig, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        init_input_shapes(inputShapes);

        bool transpA = transpose.first;
        bool transpB = transpose.second;

        fullyConnectedCount = 2;
        if (transpA)
            transposeCount += 2;
        if (!transpB)
            transposeCount++;

        if (transpA) {
            transpose_shape(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose_shape(shapes[0]);
            }
            transpose_shape(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose_shape(shapes[1]);
            }
        }
        if (transpB) {
            transpose_shape(inputDynamicShapes[2]);
            for (auto& shapes : targetStaticShapes) {
                transpose_shape(shapes[2]);
            }
        }

        const auto& inShapeFC0 = inputDynamicShapes[0];
        const auto& inShapeFC1 = inputDynamicShapes[1];
        const auto& inShapeWeights = inputDynamicShapes[2];

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ElementType netType = ElementType::f32;
        ElementType convertOutType = ElementType::f32;
        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
            convertOutType = inType = outType = netType = ElementType::bf16;
            weiConstElemType = (weiConstElemType != ElementType::f32) ? weiConstElemType : ElementType::bf16;
        } else {
            inType = outType = netType;
        }

        std::string cpuNodeType = "FullyConnected";
        selectedType = makeSelectedTypeStr(selectedType, outType);

        ov::ParameterVector params;
        for (auto&& shape : {inShapeFC0, inShapeFC1}) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        std::shared_ptr<ov::Node> inputWeights = ov::test::utils::make_constant(weiConstElemType, inShapeWeights.get_shape());
        if (weiConstElemType == ElementType::f16) {
            inputWeights = std::make_shared<ov::op::v0::Convert>(inputWeights, convertOutType);
            mark_as_decompression(inputWeights);
        }
        expectedWeiConstElemType = weiConstElemType;

        auto matMul0 = std::make_shared<ov::op::v0::MatMul>(params[0], inputWeights, transpA, transpB);
        auto matMul1 = std::make_shared<ov::op::v0::MatMul>(params[1], inputWeights, transpA, transpB);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{matMul0, matMul1}, 0);

        function = CPUTestsBase::makeNgraphFunction(netType, params, concat, cpuNodeType);
    }
};

TEST_P(MatMulDecompressConvertTest2, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_execution_graph();
}

namespace {

const auto testParams2D_FP16_2_smoke =
    ::testing::Combine(::testing::Values(static_shapes_to_test_representation({{2, 3}, {2, 3}, {3, 4}})),
                       ::testing::Values(std::pair<bool, bool>{false, true}),
                       ::testing::Values(ElementType::f16),
                       ::testing::Values(emptyConfig),
                       ::testing::ValuesIn(filter_specific_params(false)));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_FP16_2,
                         MatMulDecompressConvertTest2,
                         testParams2D_FP16_2_smoke,
                         MatMulDecompressConvertTest2::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
