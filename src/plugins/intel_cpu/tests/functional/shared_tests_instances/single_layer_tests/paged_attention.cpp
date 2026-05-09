// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/test_constants.hpp"

#include "single_op_tests/paged_attention.hpp"
#include "shared_test_classes/single_op/paged_attention.hpp"

#include "internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/core/visibility.hpp"

namespace {
using ov::test::PagedAttentionLayerTest;
using ElementType = ov::element::Type_t;
using InputShapes = std::vector<ov::test::InputShape>;

const std::vector<InputShapes> input_shapes_ref = {
{
    // Shape per step: [L, B=1, H, S] (B must be 1; metadata constants are scalar)
    {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}}},
    {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}}},
}};

// Tiny dimensions for manual verification (H=2, S=4)
const std::vector<InputShapes> input_shapes_tiny = {
{
    {{-1, 1, 2, 4}, {{3, 1, 2, 4}, {1, 1, 2, 4}}},
    {{-1, 1, 2, 4}, {{0, 1, 2, 4}, {3, 1, 2, 4}}},
}};

// Medium dimensions for alibi verification (H=4, S=8)
const std::vector<InputShapes> input_shapes_medium = {
{
    {{-1, 1, 4, 8}, {{4, 1, 4, 8}, {1, 1, 4, 8}}},
    {{-1, 1, 4, 8}, {{0, 1, 4, 8}, {4, 1, 4, 8}}},
}};

// 3-step: prompt=2 tokens, then 3 tokens (past=2, non-trivial), then decode=1 token
const std::vector<InputShapes> input_shapes_3step_tiny = {
{
    {{-1, 1, 2, 4}, {{2, 1, 2, 4}, {3, 1, 2, 4}, {1, 1, 2, 4}}},
    {{-1, 1, 2, 4}, {{0, 1, 2, 4}, {2, 1, 2, 4}, {5, 1, 2, 4}}},
}};

const std::vector<ov::AnyMap> additional_configs_ref = {{
    {ov::intel_cpu::enable_sage_attn.name(), false},

    // Force float cache (match compute precision)
    {ov::hint::kv_cache_precision.name(), ov::element::f32},
    {ov::key_cache_precision.name(), ov::element::f32},
    {ov::value_cache_precision.name(), ov::element::f32},

    // Disable grouped / quantized cache paths
    {ov::key_cache_group_size.name(), 0},
    {ov::value_cache_group_size.name(), 0},
}};

#ifdef OPENVINO_ARCH_X86_64

// Basic verification tests

// 0) Default test
INSTANTIATE_TEST_SUITE_P(smoke_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_ref),
                                            ::testing::Values(true, false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),      // sliding_window
                                            ::testing::Values(false),  // useAlibi
                                            ::testing::Values(1024),   // maxContextLen (effectively unlimited)
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 1) Tiny basic: H=2, S=4, no extras
INSTANTIATE_TEST_SUITE_P(tiny_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 2) Tiny + ALiBi
INSTANTIATE_TEST_SUITE_P(tinyAlibi_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(true),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 3) Tiny + sliding window=2, no alibi
INSTANTIATE_TEST_SUITE_P(tinySliding_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(2),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 4) Medium + ALiBi: H=4, S=8
INSTANTIATE_TEST_SUITE_P(mediumAlibi_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_medium),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(true),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// Advanced tests - all 26 inputs with various feature combinations

// 5) 3-step with non-trivial past_lens: prompt(L=2), prefill(L=3, past=2), decode(L=1)
INSTANTIATE_TEST_SUITE_P(adv3Step_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_3step_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 6) 3-step + ALiBi: tests non-trivial past_lens combined with alibi bias
INSTANTIATE_TEST_SUITE_P(adv3StepAlibi_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_3step_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(true),   // alibi
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// NOTE: max_context_len clipping is not tested because CPU ignores it during
// prefill while TEMPLATE always applies it, so there is no stable ground truth.

// Feature tests - sinks, rotation, xattention

// 7) Tiny + attention sinks
INSTANTIATE_TEST_SUITE_P(advSinks_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(true),   // sinkInput = true
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 8) Tiny + sinks + ALiBi
INSTANTIATE_TEST_SUITE_P(advSinksAlibi_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(true),   // sinkInput = true
                                            ::testing::Values(0),
                                            ::testing::Values(true),   // alibi
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// 9) Rotation: RoPE re-rotation on block 0
const std::vector<ov::AnyMap> additional_configs_rotation = {{
    {ov::intel_cpu::enable_sage_attn.name(), false},
    {ov::hint::kv_cache_precision.name(), ov::element::f32},
    {ov::key_cache_precision.name(), ov::element::f32},
    {ov::value_cache_precision.name(), ov::element::f32},
    {ov::key_cache_group_size.name(), 0},
    {ov::value_cache_group_size.name(), 0},
    {"test_use_rotation", true},
}};

INSTANTIATE_TEST_SUITE_P(advRotation_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_rotation)),
                         PagedAttentionLayerTest::getTestCaseName);

// 10) Xattention smoke: trivial 1x1 mask, validates the code path runs
INSTANTIATE_TEST_SUITE_P(advXattn_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(true),   // enableXattn = true
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_ref)),
                         PagedAttentionLayerTest::getTestCaseName);

// Adaptive RKV diversity (output not compared, just compile/crash verification)

// L=64, block_size=32, eviction_size=32 (one eviction block)
const std::vector<InputShapes> input_shapes_arkv = {
{
    {{-1, 1, 2, 4}, {{64, 1, 2, 4}, {1, 1, 2, 4}}},
    {{-1, 1, 2, 4}, {{0, 1, 2, 4}, {64, 1, 2, 4}}},
}};

const std::vector<ov::AnyMap> additional_configs_arkv = {{
    {ov::intel_cpu::enable_sage_attn.name(), false},
    {ov::hint::kv_cache_precision.name(), ov::element::f32},
    {ov::key_cache_precision.name(), ov::element::f32},
    {ov::value_cache_precision.name(), ov::element::f32},
    {ov::key_cache_group_size.name(), 0},
    {ov::value_cache_group_size.name(), 0},
    {"test_adaptive_rkv_eviction_size", 32},
}};

// 11) Adaptive RKV diversity: output 0 must still match CPU
INSTANTIATE_TEST_SUITE_P(advAdaptiveRKV_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(input_shapes_arkv),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_arkv)),
                         PagedAttentionLayerTest::getTestCaseName);

// f16/bf16 precision tests

const std::vector<ov::AnyMap> additional_configs_f16 = {{
    {ov::intel_cpu::enable_sage_attn.name(), false},
    {ov::hint::kv_cache_precision.name(), ov::element::f16},
    {ov::key_cache_precision.name(), ov::element::f16},
    {ov::value_cache_precision.name(), ov::element::f16},
    {ov::key_cache_group_size.name(), 0},
    {ov::value_cache_group_size.name(), 0},
    {"test_abs_threshold", 0.05f},
    {"test_rel_threshold", 0.1f},
}};

const std::vector<ov::AnyMap> additional_configs_bf16 = {{
    {ov::intel_cpu::enable_sage_attn.name(), false},
    {ov::hint::kv_cache_precision.name(), ov::element::bf16},
    {ov::key_cache_precision.name(), ov::element::bf16},
    {ov::value_cache_precision.name(), ov::element::bf16},
    {ov::key_cache_group_size.name(), 0},
    {ov::value_cache_group_size.name(), 0},
    {"test_abs_threshold", 0.05f},
    {"test_rel_threshold", 0.1f},
}};

// 12) f16: basic tiny test
INSTANTIATE_TEST_SUITE_P(f16_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::f16),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_f16)),
                         PagedAttentionLayerTest::getTestCaseName);

// 13) bf16: basic tiny test
INSTANTIATE_TEST_SUITE_P(bf16_PagedAttentionLayerTest,
                         PagedAttentionLayerTest,
                         ::testing::Combine(::testing::Values(ElementType::bf16),
                                            ::testing::ValuesIn(input_shapes_tiny),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(false),
                                            ::testing::Values(1024),
                                            ::testing::ValuesIn(additional_configs_bf16)),
                         PagedAttentionLayerTest::getTestCaseName);

#endif  // OPENVINO_ARCH_X86_64
}  // namespace
