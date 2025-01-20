// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> transposedShape_4D(bool with_static = true, bool with_dynamic = true) {
    std::vector<std::vector<ov::test::InputShape>> shapes;
    if (with_static) {
        auto static_shapes =
            SNIPPETS_TESTS_STATIC_SHAPES({{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
                                         {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}},
                                         {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
                                         {{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}},
                                         {{1, 58, 16, 34}, {1, 58, 16, 34}, {1, 1, 1, 58}, {1, 58, 16, 34}});
        shapes.insert(shapes.end(), static_shapes.begin(), static_shapes.end());
    }
    if (with_dynamic) {
        std::vector<std::vector<ov::test::InputShape>> dynamic_shapes = {
            {
                {PartialShape{-1, -1, -1, 100}, {{1, 64, 4, 100}, {2, 16, 2, 100}, {1, 72, 4, 100}}},
                {PartialShape{-1, 128, -1, 100}, {{1, 128, 4, 100}, {2, 128, 2, 100}, {1, 128, 4, 100}}},
                {PartialShape{-1, -1, -1, 128}, {{1, 4, 64, 128}, {2, 2, 16, 128}, {1, 4, 72, 128}}},
                {PartialShape{-1, 128, -1, 100}, {{1, 128, 4, 100}, {2, 128, 2, 100}, {1, 128, 4, 100}}},
            },
            {
                {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {2, 16, 2, 100}, {1, 128, 3, 64}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 128, 1, 64}, {2, 128, 2, 100}, {1, 128, 1, 64}}},
                {PartialShape{-1, -1, -1, -1}, {{2, 1, 128, 128}, {2, 2, 16, 128}, {2, 1, 128, 128}}},
                {PartialShape{-1, -1, -1, -1}, {{1, 128, 3, 64}, {2, 128, 2, 100}, {1, 128, 3, 64}}},
            },
            {
                {PartialShape{-1, -1, 12, 64},
                 {{1, 70, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 20, 12, 64}, {1, 70, 12, 64}}},
                {PartialShape{-1, -1, 12, 64},
                 {{1, 35, 12, 64}, {2, 10, 12, 64}, {2, 1, 12, 64}, {2, 10, 12, 64}, {1, 35, 12, 64}}},
                {PartialShape{-1, 12, -1, -1},
                 {{2, 12, 70, 35}, {1, 12, 20, 10}, {1, 12, 20, 10}, {1, 12, 20, 1}, {2, 12, 70, 35}}},
                {PartialShape{-1, -1, 12, 64},
                 {{1, 35, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 10, 12, 64}, {1, 35, 12, 64}}},
            }};
        shapes.insert(shapes.end(), dynamic_shapes.begin(), dynamic_shapes.end());
    }
    return shapes;
}

std::vector<std::vector<InputShape>> transposedShape_3D(bool with_dynamic = true) {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{128, 12, 64}, {128, 12, 64}, {12, 128, 128}, {128, 12, 64}},
        {{68, 6, 92}, {68, 6, 92}, {1, 68, 68}, {68, 6, 92}},
        {{16, 2, 92}, {68, 2, 92}, {1, 16, 68}, {68, 2, 92}});
    if (with_dynamic) {
        shapes.push_back({
            {PartialShape{-1, -1, -1}, {{128, 3, 64},  {128, 3, 64},  {68, 6, 87}}},
            {PartialShape{-1, -1, -1}, {{128, 1, 64},  {128, 1, 64},  {13, 6, 87}}},
            {PartialShape{-1, -1, -1}, {{1, 128, 128}, {1, 128, 128}, {1, 68, 13}}},
            {PartialShape{-1, -1, -1}, {{128, 3, 64},  {128, 3, 64},  {13, 6, 87}}},
        });
    }
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_4D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D()),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(2), // decomposed Transpose + MHA
                                            ::testing::Values(2), // decomposed Transpose + MHA
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_4D_WithScalarMul,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D(true, false)),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(true),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(2), // decomposed Transpose + MHA
                                            ::testing::Values(2), // decomposed Transpose, Mul + MHA
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_3D()),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(5),  // [122706]: Subgraph + 4 Transpose
                                            ::testing::Values(2),  // decomposed Transpose + MHA
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_3D_WithScalarMul,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_3D(false)),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(true),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(5),  // [122706]: Subgraph + 4 Transpose
                                            ::testing::Values(2),  // decomposed Transpose + MHA
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHABF16_4D,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D()),
                                            ::testing::ValuesIn(precision_bf16_if_supported(4)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(8),  // decomposed Transpose + MHA + 5 Converts + 1 Transpose on output
                                            ::testing::Values(6),  // MHA + 5 Converts on inputs and output
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceBF16,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D()),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::ValuesIn({false}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(8),  // decomposed Transpose + MHA + 5 Converts + 1 Transpose on output
                                            ::testing::Values(6),  // MHA + 5 Reorders on inputs and output
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::cpu_bf16_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_FP16_4D_Without_Multiply,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D()),
                                            ::testing::ValuesIn(precision_fp16_if_supported(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({false}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(3),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_FP16_4D_With_Multiply_Static,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D(true, false)),
                                            ::testing::ValuesIn(precision_fp16_if_supported(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(3),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);
// 3 nodes and 2 subgraph for dynamic with multiply case.
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_FP16_4D_With_Multiply_Dynamic,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D(false, true)),
                                            ::testing::ValuesIn(precision_fp16_if_supported(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(4),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceFP16_Without_Multiply,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D()),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({false}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(3),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::cpu_f16_plugin_config)),
                         MHA::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceFP16_With_Multiply_Static,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D(true, false)),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(3),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::cpu_f16_plugin_config)),
                         MHA::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAEnforceFP16_With_Multiply_Dynamic,
                         MHA,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_4D(false, true)),
                                            ::testing::ValuesIn(precision_f32(4)),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn({true}),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(4),
                                            ::testing::Values(2),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::cpu_f16_plugin_config)),
                         MHA::getTestCaseName);
}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
