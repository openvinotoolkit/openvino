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

const std::vector<InputShapes> input_shapes_ref = {  // greedy search
{
    // L, B=1, H, S   (B must be 1: scalar metadata constants in the PA model
    //                  are only valid for single-sequence operation)
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

// ═══════════════════════════════════════════════════════════
//  Smoke tests (original)
// ═══════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════
//  Tiny verification tests
// ═══════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════
//  Advanced tests — exercise previously-untested inputs
// ═══════════════════════════════════════════════════════════

// 5) 3-step with non-trivial past_lens: L=2/3/1
//    Step 0: past=0, L=2  (prompt)
//    Step 1: past=2, L=3  (past is between 0 and L!)
//    Step 2: past=5, L=1  (decode)
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

// 6) 3-step + ALiBi — non-trivial past_lens combined with alibi.
//    Exercises the alibi bias computation when the context grows
//    across three separate inference steps (past=0/2/5).
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

// NOTE: max_context_len clipping (inputs 12) cannot be tested here because
// the CPU kernel's exec_loop_mixed path declares max_context_len as
// [[maybe_unused]] and ignores it during multi-token prefill, while the
// TEMPLATE reference applies per-token clipping.  This is a known semantic
// difference — testing would always diverge when clipping is triggered.

// ═══════════════════════════════════════════════════════════
//  Feature tests — sinks, rotation, xattention
// ═══════════════════════════════════════════════════════════

// 7) Tiny + attention sinks: [1,H,1,1] per-head logit adds virtual token
//    to softmax denominator.  Verifies that both CPU and reference apply
//    the same sink contribution.
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

// 8) Tiny + sinks + ALiBi: exercises sinks combined with alibi bias.
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

// 9) Rotation test: 2-step tiny test (prefill+decode) with cache rotation
//    enabled on block 0.  Step 0 populates cache; step 1 reads rotated cache.
//    The rotation flag is passed via the config map.
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

// 10) Xattention smoke test: exercises the dynamic sparse attention code path.
//     With tiny shapes (L=3, xattention_block_size=64), the mask is 1×1
//     (trivially [[true]]), so no blocks are masked.  The test validates that
//     both CPU and reference run the xattention path without errors and
//     produce the same output.
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

#endif  // OPENVINO_ARCH_X86_64
}  // namespace


