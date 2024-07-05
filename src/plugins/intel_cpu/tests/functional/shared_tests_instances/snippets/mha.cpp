// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "common_test_utils/test_constants.hpp"
#include "internal_properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {
namespace snippets {

#define STATIC_SHAPE(...) {{}, {{__VA_ARGS__}}}

namespace {

const std::vector<std::vector<ov::test::InputShape>> inputShapes_4D = {
    {STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 12, 128, 128), STATIC_SHAPE(1, 128, 12, 64)},
    {STATIC_SHAPE(1, 128, 16, 64), STATIC_SHAPE(1, 128, 16, 64), STATIC_SHAPE(1, 16, 1, 1), STATIC_SHAPE(1, 128, 16, 64)},
    {STATIC_SHAPE(1, 128, 16, 64), STATIC_SHAPE(1, 128, 16, 64), STATIC_SHAPE(1, 1, 1, 128), STATIC_SHAPE(1, 128, 16, 64)},
    {STATIC_SHAPE(2, 68, 6, 92), STATIC_SHAPE(2, 68, 6, 92), STATIC_SHAPE(1, 1, 68, 68), STATIC_SHAPE(2, 68, 6, 92)},
    {STATIC_SHAPE(1, 58, 16, 34), STATIC_SHAPE(1, 58, 16, 34), STATIC_SHAPE(1, 1, 1, 58), STATIC_SHAPE(1, 58, 16, 34)},
};

const std::vector<std::vector<InputShape>> inputShapes_3D = {
    {STATIC_SHAPE(128, 12, 64), STATIC_SHAPE(128, 12, 64), STATIC_SHAPE(12, 128, 128), STATIC_SHAPE(128, 12, 64)},
    {STATIC_SHAPE(68, 6, 92), STATIC_SHAPE(68, 6, 92), STATIC_SHAPE(1, 68, 68), STATIC_SHAPE(68, 6, 92)},
};

static inline bool is_bf16_supported() {
    return ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx_bf16();
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

static ov::AnyMap enable_callback() {
    return ov::AnyMap({ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::ENABLE)});
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_4D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn({false, true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> inputShapes_4D_dynamic{
        {
            {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 70, 3, 19}, {1, 128, 3, 64}, {1, 68, 6, 87}}},
            {PartialShape{-1, -1, -1, -1}, {{1, 128, 1, 64}, {2, 49, 1, 19}, {1, 128, 1, 64}, {2, 13, 6, 87}}},
            {PartialShape{-1, -1, -1, -1}, {{2, 1, 128, 128}, {1, 1, 70, 49}, {2, 1, 128, 128}, {1, 1, 68, 13}}},
            {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {1, 49, 3, 19}, {1, 128, 3, 64}, {2, 13, 6, 87}}},
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_DynMHA_4D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D_dynamic),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn({false}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_3D),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn({false, true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(5),  // [122706]: Subgraph + 4 Transpose
                                            ::testing::Values(2),  // decomposed Transpose + MHA
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_4D_SplitDimensionM,
    MHA,
    ::testing::Combine(::testing::Values(std::vector<InputShape>{STATIC_SHAPE(1, 128, 2, 64),
                                                                 STATIC_SHAPE(1, 128, 2, 64),
                                                                 STATIC_SHAPE(1, 1, 1, 1),
                                                                 STATIC_SHAPE(1, 128, 2, 64)}),
                       ::testing::ValuesIn(precision_f32(4)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),
                       ::testing::Values(4),  // 4 Threads
                       ::testing::Values(6),  // Subgraph + 4 Reshapes on inputs and 1 Reshape on output
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(enable_callback())),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA_3D_SplitDimensionM,
    MHA,
    ::testing::Combine(
        ::testing::Values(std::vector<InputShape>{STATIC_SHAPE(384, 2, 64),
                                                  STATIC_SHAPE(384, 2, 64),
                                                  STATIC_SHAPE(1, 384, 384),
                                                  STATIC_SHAPE(384, 2, 64)}),
        ::testing::ValuesIn(precision_f32(4)),
        ::testing::Values(ov::element::f32),
        ::testing::Values(true),
        ::testing::Values(4),   // 4 Threads
        ::testing::Values(10),  // Subgraph + 4 Reshapes on inputs and 1 Reshape on output + 4 Transposes
        ::testing::Values(1),   // MHA
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(enable_callback())),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHABF16_4D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D),
                                            ::testing::ValuesIn(precision_bf16(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn({false, true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(7),  // MHA + 5 Converts + 1 Transpose on output
                                            ::testing::Values(6),  // MHA + 5 Converts on inputs and output
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceBF16,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::ValuesIn({false}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(7),
                                            ::testing::Values(6),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::cpu_bf16_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAMulAdd,
    MHAMulAdd,
    ::testing::Combine(
        ::testing::Values(std::vector<InputShape>{STATIC_SHAPE(1, 10, 12, 16),
                                                  STATIC_SHAPE(1, 10, 12, 16),
                                                  STATIC_SHAPE(1, 10, 12, 16)}),
        ::testing::ValuesIn(precision_f32(3)),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn({false}),  // Need to support True for graph builder in tests
        ::testing::Values(MHA::default_thread_count),
        ::testing::Values(1),
        ::testing::Values(1),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapeSelect = {
    // without broadcast
    {STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 12, 128, 128),
     STATIC_SHAPE(1, 12, 128, 128), STATIC_SHAPE(1, 12, 128, 128), STATIC_SHAPE(1, 128, 12, 64)},
    {STATIC_SHAPE(1, 94, 12, 54), STATIC_SHAPE(1, 94, 12, 54), STATIC_SHAPE(1, 12, 94, 94),
     STATIC_SHAPE(1, 12, 94, 94), STATIC_SHAPE(1, 12, 94, 94), STATIC_SHAPE(1, 94, 12, 54)},
    // with broadcast
    {STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 128, 12, 64), STATIC_SHAPE(1, 12, 128, 128),
     STATIC_SHAPE(1, 12, 1, 1), STATIC_SHAPE(1, 12, 1, 1), STATIC_SHAPE(1, 128, 12, 64)},
    {STATIC_SHAPE(2, 52, 6, 102), STATIC_SHAPE(2, 52, 6, 102), STATIC_SHAPE(1, 6, 52, 52),
     STATIC_SHAPE(1, 6, 1, 1), STATIC_SHAPE(1, 6, 1, 1), STATIC_SHAPE(2, 52, 6, 102)}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA,
    MHASelect,
    ::testing::Combine(::testing::ValuesIn(inputShapeSelect),
                       ::testing::ValuesIn(precision_f32(6)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // Need to support True for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(2),  // Less + MHA
                       ::testing::Values(2),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapesWOTranspose_4D = {
    {STATIC_SHAPE(1, 12, 197, 64), STATIC_SHAPE(1, 12, 64, 197), STATIC_SHAPE(1, 12, 197, 64)},
    {STATIC_SHAPE(1, 12, 12, 64), STATIC_SHAPE(1, 12, 64, 48), STATIC_SHAPE(1, 12, 48, 64)}};
const std::vector<std::vector<InputShape>> inputShapesWOTranspose_3D = {
    {STATIC_SHAPE(12, 197, 64), STATIC_SHAPE(12, 64, 197), STATIC_SHAPE(12, 197, 64)},
    {STATIC_SHAPE(12, 128, 100), STATIC_SHAPE(12, 100, 128), STATIC_SHAPE(12, 128, 100)}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeOnInputs_4D,
    MHAWOTransposeOnInputs,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_4D),
                       ::testing::Values(std::vector<ov::element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(true),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTranspose_4D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_4D),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTranspose_3D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_3D),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeBF16_4D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_4D),
                       ::testing::ValuesIn(precision_bf16(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeBF16_3D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_3D),
                       ::testing::ValuesIn(precision_bf16(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeEnforceBF16_4D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_4D),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::bf16),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::cpu_bf16_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWOTransposeEnforceBF16_3D,
    MHAWOTranspose,
    ::testing::Combine(::testing::ValuesIn(inputShapesWOTranspose_3D),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::bf16),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(5),  // MHA + 4 extra Converts on inputs and output
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::cpu_bf16_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAINT8MatMul,
    MHAINT8MatMul,
    ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>(inputShapes_4D.begin(),
                                                                                      inputShapes_4D.begin() + 2)),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // The graph doesn't contain Multiply
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(6),  // FQx3 on inputs + MHA + Transpose on output + Deq Mul
                       ::testing::Values(5),  // FQx3 on inputs + MHA + Deq Mul
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAQuantMatMul0,
    MHAQuantMatMul0,
    ::testing::Combine(
        ::testing::Values(std::vector<InputShape>{STATIC_SHAPE(1, 128, 768), STATIC_SHAPE(1, 128, 768), STATIC_SHAPE(1, 1, 1, 128), STATIC_SHAPE(1, 128, 768)}),
        ::testing::Values(std::vector<element::Type>{}),
        ::testing::Values(ov::element::f32),
        ::testing::Values(false),  // The graph doesn't contain Multiply
        ::testing::Values(MHA::default_thread_count),
        ::testing::Values(9),  // FQx2 on inputs + MHA + Transpose on output + 4 Reshapes + Deq Mul
        ::testing::Values(4),  // FQx2 on inputs + MHA + Deq Mul
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAFQAfterMatMul_4D,
                         MHAFQAfterMatMul,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_4D),
                                            ::testing::Values(std::vector<element::Type>{}),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),  // The graph doesn't contain Multiply
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(3),  // MHA + Transpose on output + Deq Mul
                                            ::testing::Values(2),  // MHA + Deq Mul
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAFQ,
    MHAFQ,
    ::testing::Combine(::testing::Values(std::vector<InputShape>{STATIC_SHAPE(1, 64, 12, 64),
                                                                 STATIC_SHAPE(1, 64, 12, 64),
                                                                 STATIC_SHAPE(1, 1, 1, 64),
                                                                 STATIC_SHAPE(1, 64, 12, 64)}),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // The graph doesn't contain Multiply
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(7),  // Transposex2 + Subgraphsx5
                       ::testing::Values(5),  // MHA + Deq Mul on output + Deqs on inputs + 2 xFQ on inputs
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapesTransposedB = {
    {STATIC_SHAPE(1, 12, 12, 64), STATIC_SHAPE(1, 12, 48, 64), STATIC_SHAPE(1, 12, 48, 64)}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHATransposedB,
    MHATransposedB,
    ::testing::Combine(::testing::ValuesIn(inputShapesTransposedB),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(2),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

const std::vector<std::vector<InputShape>> inputShapesExtractedReshape = {
    {STATIC_SHAPE(2, 196, 64), STATIC_SHAPE(2, 64, 196), STATIC_SHAPE(2, 14, 14, 14, 1), STATIC_SHAPE(2, 14, 14, 1, 14), STATIC_SHAPE(2, 196, 64)},
    {STATIC_SHAPE(1, 16, 10), STATIC_SHAPE(1, 10, 16), STATIC_SHAPE(1, 4, 4, 4, 1), STATIC_SHAPE(1, 4, 4, 1, 4), STATIC_SHAPE(1, 16, 10)},
    {STATIC_SHAPE(1, 16, 10), STATIC_SHAPE(1, 10, 16), STATIC_SHAPE(1, 1, 1, 1, 1), STATIC_SHAPE(1, 4, 4, 4, 4), STATIC_SHAPE(1, 16, 10)},
    {STATIC_SHAPE(1, 16, 10), STATIC_SHAPE(1, 10, 16), STATIC_SHAPE(1, 4, 4, 4, 4), STATIC_SHAPE(1, 1, 1, 1, 1), STATIC_SHAPE(1, 16, 10)},
    {STATIC_SHAPE(1, 4, 16, 10), STATIC_SHAPE(1, 4, 10, 16), STATIC_SHAPE(1, 4, 256), STATIC_SHAPE(1, 4, 256), STATIC_SHAPE(1, 4, 16, 10)},
    {STATIC_SHAPE(1, 4, 16, 10), STATIC_SHAPE(1, 4, 10, 16), STATIC_SHAPE(1, 1, 256), STATIC_SHAPE(1, 4, 1), STATIC_SHAPE(1, 4, 16, 10)},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWithExtractedReshape,
    MHAWithExtractedReshape,
    ::testing::Combine(::testing::ValuesIn(inputShapesExtractedReshape),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // False is not supported for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(3),  // Extracted Add + Extracted Reshape + MHA
                       ::testing::Values(2),  // Extracted Add + MHA
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
