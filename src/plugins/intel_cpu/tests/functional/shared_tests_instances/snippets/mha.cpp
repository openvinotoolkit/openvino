// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "ie_plugin_config.hpp"
#include "ie_system_conf.h"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> inputShapes_4D = {
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
        {{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}},
        {{1, 58, 16, 34}, {1, 58, 16, 34}, {1, 1, 1, 58}, {1, 58, 16, 34}},
};

const std::vector<std::vector<ov::PartialShape>> inputShapes_3D = {
        {{128, 12, 64}, {128, 12, 64}, {12, 128, 128}, {128, 12, 64}},
        {{68, 6, 92}, {68, 6, 92}, {1, 68, 68}, { 68, 6, 92}},
};

static inline bool is_bf16_supported() {
    return InferenceEngine::with_cpu_x86_bfloat16() || InferenceEngine::with_cpu_x86_avx512_core_amx_bf16();
}

static inline std::vector<std::vector<element::Type>> precision_f32(size_t count) {
    std::vector<std::vector<element::Type>> prc;
    prc.emplace_back(std::vector<element::Type>(count, element::f32));
    return prc;
}

static inline std::vector<std::vector<element::Type>> precision_bf16(size_t count) {
    std::vector<std::vector<element::Type>> prc;
    if (is_bf16_supported())
        prc.emplace_back(std::vector<element::Type>(count, element::bf16));
    return prc;
}

static std::map<std::string, std::string> enable_callback() {
    return std::map<std::string, std::string>{
        {
          InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
          InferenceEngine::PluginConfigInternalParams::ENABLE
        },
     };
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_4D, MHA,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_4D),
                                 ::testing::ValuesIn(precision_f32(4)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({false, true}),
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D, MHA,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_3D),
                                 ::testing::ValuesIn(precision_f32(4)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({false, true}),
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(5), // [122706]: Subgraph + 4 Transpose
                                 ::testing::Values(2), // decomposed Transpose + MHA
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_4D_SplitDimensionM, MHA,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 128, 2, 64}, {1, 128, 2, 64}, {1, 1, 1, 1}, {1, 128, 2, 64}}),
                                 ::testing::ValuesIn(precision_f32(4)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(true),
                                 ::testing::Values(4), // 4 Threads
                                 ::testing::Values(6), // Subgraph + 4 Reshapes on inputs and 1 Reshape on output
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(enable_callback())),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D_SplitDimensionM, MHA,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{384, 2, 64}, {384, 2, 64}, {1, 384, 384}, {384, 2, 64}}),
                                 ::testing::ValuesIn(precision_f32(4)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(true),
                                 ::testing::Values(4), // 4 Threads
                                 ::testing::Values(10), // Subgraph + 4 Reshapes on inputs and 1 Reshape on output + 4 Transposes
                                 ::testing::Values(1), // MHA
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(enable_callback())),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHABF16_4D, MHA,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_4D),
                                 ::testing::ValuesIn(precision_bf16(4)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({false, true}),
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(7), // MHA + 5 Converts + 1 Transpose on output
                                 ::testing::Values(6), // MHA + 5 Converts on inputs and output
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceBF16, MHA,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_4D),
                                 ::testing::ValuesIn(precision_f32(4)),
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::ValuesIn({false}),
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(7),
                                 ::testing::Values(7),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuBF16PluginConfig)),
                         MHA::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAMulAdd, MHAMulAdd,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 10, 12, 16}, {1, 10, 12, 16}, {1, 10, 12, 16}}),
                                 ::testing::ValuesIn(precision_f32(3)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({false}),  // Need to support True for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(std::map<std::string, std::string>{})),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapeSelect = {
        // without broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
        // with broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
        {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHASelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeSelect),
                                 ::testing::ValuesIn(precision_f32(6)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),  // Need to support True for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(2), // Less + MHA
                                 ::testing::Values(2),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose_4D = {
        {{1, 12, 197, 64}, {1, 12, 64, 197}, {1, 12, 197, 64}},
        {{1, 12, 12, 64}, {1, 12, 64, 48}, {1, 12, 48, 64}}
};
const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose_3D = {
        {{12, 197, 64}, {12, 64, 197}, {12, 197, 64}},
        {{12, 128, 100}, {12, 100, 128}, {12, 128, 100}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputs_4D, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_4D),
                                 ::testing::Values(std::vector<ov::element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(true),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTranspose_4D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_4D),
                                 ::testing::ValuesIn(precision_f32(3)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTranspose_3D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_3D),
                                 ::testing::ValuesIn(precision_f32(3)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeBF16_4D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_4D),
                                 ::testing::ValuesIn(precision_bf16(3)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeBF16_3D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_3D),
                                 ::testing::ValuesIn(precision_bf16(3)),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeEnforceBF16_4D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_4D),
                                 ::testing::ValuesIn(precision_f32(3)),
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuBF16PluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeEnforceBF16_3D, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose_3D),
                                 ::testing::ValuesIn(precision_f32(3)),
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(5), // MHA + 4 extra Converts on inputs and output
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuBF16PluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAINT8MatMul, MHAINT8MatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<std::vector<ov::PartialShape>>(inputShapes_4D.begin(), inputShapes_4D.begin() + 2)),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(6),     // FQx3 on inputs + MHA + Transpose on output + Deq Mul
                                 ::testing::Values(5),     // FQx3 on inputs + MHA + Deq Mul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAQuantMatMul0, MHAQuantMatMul0,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 128, 768}, {1, 128, 768}, {1, 1, 1, 128}, {1, 128, 768}}),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(8),     // FQ on input + MHA + Transpose on output + 4 Reshapes + Deq Mul
                                 ::testing::Values(3),     // FQ on input + MHA + Deq Mul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAFQAfterMatMul_4D, MHAFQAfterMatMul,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_4D),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(3),     // MHA + Transpose on output + Deq Mul
                                 ::testing::Values(2),     // MHA + Deq Mul
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAFQ, MHAFQ,
                         ::testing::Combine(
                                 ::testing::Values(std::vector<ov::PartialShape>{{1, 64, 12, 64}, {1, 64, 12, 64}, {1, 1, 1, 64}, {1, 64, 12, 64}}),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false), // The graph doesn't contain Multiply
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(7),     // Transposex2 + Subgraphsx5
                                 ::testing::Values(5),     // MHA + Deq Mul on output + Deqs on inputs + 2 xFQ on inputs
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesTransposedB = {
        {{1, 12, 12, 64}, {1, 12, 48, 64}, {1, 12, 48, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHATransposedB, MHATransposedB,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesTransposedB),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(2),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(std::map<std::string, std::string>{})),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesExtractedReshape = {
    {{2, 196, 64}, {2, 64, 196}, {2, 14, 14, 14, 1}, {2, 14, 14, 1, 14}, {2, 196, 64}},
    {{1, 16, 10}, {1, 10, 16}, {1, 4, 4, 4, 1}, {1, 4, 4, 1, 4}, {1, 16, 10}},
    {{1, 16, 10}, {1, 10, 16}, {1, 1, 1, 1, 1}, {1, 4, 4, 4, 4}, {1, 16, 10}},
    {{1, 16, 10}, {1, 10, 16}, {1, 4, 4, 4, 4}, {1, 1, 1, 1, 1}, {1, 16, 10}},
    {{1, 4, 16, 10}, {1, 4, 10, 16}, {1, 4, 256}, {1, 4, 256}, {1, 4, 16, 10}},
    {{1, 4, 16, 10}, {1, 4, 10, 16}, {1, 1, 256}, {1, 4, 1}, {1, 4, 16, 10}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWithExtractedReshape, MHAWithExtractedReshape,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesExtractedReshape),
                                 ::testing::Values(std::vector<element::Type>{}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::ValuesIn({true}), // False is not supported for graph builder in tests
                                 ::testing::Values(MHA::default_thread_count),
                                 ::testing::Values(3), // Extracted Add + Extracted Reshape + MHA
                                 ::testing::Values(2), // Extracted Add + MHA
                                 ::testing::Values(ov::test::utils::DEVICE_CPU),
                                 ::testing::Values(std::map<std::string, std::string>{})),
                         MHA::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
