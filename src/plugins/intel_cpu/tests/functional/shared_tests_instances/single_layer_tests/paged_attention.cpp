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

// --- Smoke tests (original)

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

// --- Basic verification tests - first 11 inputs

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

// --- Advanced tests - all 25 inputs with various feature combinations

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

// NOTE: We skip testing max_context_len clipping (input 12) because the CPU and TEMPLATE
// handle it differently.  During multi-token prefill (when a batch of tokens is processed
// at once, e.g. the initial prompt pass), the CPU kernel ignores max_context_len, but the
// TEMPLATE reference always applies it.  Any test with active clipping would always
// produce different results between the two, so there is no stable ground truth to compare

// --- Feature tests - sinks, rotation, xattention

// 7) Tiny + attention sinks: per-head logit added as a virtual token in the softmax denominator
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

// 9) Rotation: 2-step (prefill+decode) with RoPE re-rotation enabled on block 0
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

// 10) Xattention smoke: with tiny shapes (L=3, xattn_block_size=64) the mask is trivially
//     1x1 [[true]], so no blocks are masked - validates the code path runs without error
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

// --- Adaptive RKV diversity test

// Shapes for adaptive RKV: L=64 with block_size=32 (forced by CPU plugin's ConvertPagedAttnInputs)
// and eviction_size=32 so the eviction zone spans exactly one block
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

// 11) Adaptive RKV diversity: verifies diversity scoring in the reference
//     Output 0 (attention) is unaffected by adaptive RKV and must match CPU
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

#endif  // OPENVINO_ARCH_X86_64
}  // namespace
