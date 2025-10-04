// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/paged_attention.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_op/paged_attention.hpp"

namespace {
using ov::test::PagedAttentionLayerTest;
const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16
};

//===========================================================
//    • Query/Key/Value: [2,8] → 2 tokens, 2 heads × head_size=4
//    • Key/Value Cache: [3,2,32,4] → 3 blocks, 2 heads, block_size=32, head_size=4
//    • past_lens: {0,1}, subsequence_begins: {0,1,2}
//    • block_indices: length=3, block_indices_begins: length=3
//    • Misc: scale=1, sliding_window=1, alibi_slopes={2,4}, max_context_len=1
//    • No rotation
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_basic_static,
    PagedAttentionLayerTest,
    ::testing::Combine(
        // 1) Static shapes: { query, key, value, key_cache, value_cache }
        ::testing::Values(ov::test::static_shapes_to_test_representation({
            {2, 8},          // query:      2 tokens, 2 heads × head_size=4
            {2, 8},          // key:        2 tokens, 2 heads × head_size=4
            {2, 8},          // value:      2 tokens, 2 heads × head_size=4
            {3, 2, 32, 4},   // key_cache:   3 blocks, 2 heads, block_size=32, head_size=4
            {3, 2, 32, 4}    // value_cache: 3 blocks, 2 heads, block_size=32, head_size=4
        })),
        // 2) Integer-vector inputs
        ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
            /* past_lens                */ {0, 1},
            /* subsequence_begins       */ {0, 1, 2},
            /* block_indices            */ {0, 1, 2},
            /* block_indices_begins     */ {0, 1, 2}
        }),
        // 3) Miscellaneous inputs
        ::testing::Values(ov::test::PagedAttentionMiscInpStruct{
            /* scale                   */ {1},        // fixed scale=1.0
            /* sliding_window          */ 1,          // window=1
            /* alibi_slopes            */ {2, 4},     // two slopes for two heads
            /* max_context_len         */ 1
        }),
        // 4) No rotation for this basic test
        ::testing::Values(std::nullopt),
        // 5) Precision: f32, f16, bf16
        ::testing::ValuesIn(inputPrecisions),
        // 6) Device
        ::testing::Values(ov::test::utils::DEVICE_CPU)
    ),
    PagedAttentionLayerTest::getTestCaseName
);

//===========================================================
//    • Query/Key/Value: [2,8] → 2 tokens, 2 heads × head_size=4
//    • Key/Value Cache: [1,2,32,4] → 1 block, 2 heads, block_size=32, head_size=4
//    • past_lens: {0,1}, subsequence_begins: {0,1,2}
//    • block_indices: {0}, block_indices_begins: {0,0,0}
//    • Misc: scale=1, sliding_window=0, alibi_slopes={0,0}, max_context_len=10
//    • RotationStruct:
//        – rotated_block_indices length=4
//        – rotation_deltas shape={4,32}
//        – rotation_trig_lut shape={4,4}
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_static_rotation,
    PagedAttentionLayerTest,
    ::testing::Combine(
        // 1) Static shapes
        ::testing::Values(ov::test::static_shapes_to_test_representation({
            {2, 8},          // query:      2 tokens, 2 heads × head_size=4
            {2, 8},          // key:        2 tokens, 2 heads × head_size=4
            {2, 8},          // value:      2 tokens, 2 heads × head_size=4
            {1, 2, 32, 4},   // key_cache:   1 block, 2 heads, block_size=32, head_size=4
            {1, 2, 32, 4}    // value_cache: 1 block, 2 heads, block_size=32, head_size=4
        })),
        // 2) Integer-vector inputs
        ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
            /* past_lens                */ {0, 1},
            /* subsequence_begins       */ {0, 1, 2},
            /* block_indices            */ {0},
            /* block_indices_begins     */ {0, 0, 0}
        }),
        // 3) Miscellaneous inputs
        ::testing::Values(ov::test::PagedAttentionMiscInpStruct{
            /* scale                   */ {1},        // fixed scale=1.0
            /* sliding_window          */ 0,          // no sliding-window masking
            /* alibi_slopes            */ {0, 0},     // no ALiBi bias
            /* max_context_len         */ 10
        }),
        // 4) Rotation inputs
        ::testing::Values(ov::test::PagedAttentionRotationStruct{
            /* rotated_block_indices   */ {0, 0, 0, 0},      // length=4
            /* rotation_deltas shape   */ ov::Shape{4, 32},   // 4×32
            /* rotation_trig_lut shape */ ov::Shape{4, 4}     // 4×4
        }),
        // 5) Precision: f32, f16, bf16
        ::testing::ValuesIn(inputPrecisions),
        // 6) Device
        ::testing::Values(ov::test::utils::DEVICE_CPU)
    ),
    PagedAttentionLayerTest::getTestCaseName
);

//===========================================================
//    • Query/Key/Value: [2,8] → 2 tokens, 2 heads × head_size=4
//    • Key/Value Cache: [3,2,32,4] → 3 blocks, 2 heads, block_size=32, head_size=4
//    • past_lens: {0,1}, subsequence_begins: {0,1,2}
//    • block_indices: length=3, block_indices_begins: length=3
//    • Misc: scale={1.5}, {5.0}, {} (default), sliding_window=1, alibi_slopes={2,4}, max_context_len=1
//    • No rotation
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_scale,
    PagedAttentionLayerTest,
    ::testing::Combine(
        // 1) Static shapes
        ::testing::Values(ov::test::static_shapes_to_test_representation({
            {2, 8},          // query:      2 tokens, 2 heads × head_size=4
            {2, 8},          // key:        2 tokens, 2 heads × head_size=4
            {2, 8},          // value:      2 tokens, 2 heads × head_size=4
            {3, 2, 32, 4},   // key_cache:   3 blocks, 2 heads, block_size=32, head_size=4
            {3, 2, 32, 4}    // value_cache: 3 blocks, 2 heads, block_size=32, head_size=4
        })),
        // 2) Integer-vector inputs
        ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
            /* past_lens                */ {0, 1},
            /* subsequence_begins       */ {0, 1, 2},
            /* block_indices            */ {0, 1, 2},
            /* block_indices_begins     */ {0, 1, 2}
        }),
        // 3) Miscellaneous inputs: three scale variants
        ::testing::ValuesIn({
            ov::test::PagedAttentionMiscInpStruct{
                /* scale               */ {1.5f},
                /* sliding_window      */ 1,
                /* alibi_slopes        */ {2, 4},
                /* max_context_len     */ 1
            },
            ov::test::PagedAttentionMiscInpStruct{
                /* scale               */ {5.0f},
                /* sliding_window      */ 1,
                /* alibi_slopes        */ {2, 4},
                /* max_context_len     */ 1
            },
            ov::test::PagedAttentionMiscInpStruct{
                /* scale               */ {},        // default scale
                /* sliding_window      */ 1,
                /* alibi_slopes        */ {2, 4},
                /* max_context_len     */ 1
            }
        }),
        // 4) No rotation
        ::testing::Values(std::nullopt),
        // 5) Precision: f32, f16, bf16
        ::testing::ValuesIn(inputPrecisions),
        // 6) Device
        ::testing::Values(ov::test::utils::DEVICE_CPU)
    ),
    PagedAttentionLayerTest::getTestCaseName
);

//===========================================================
//    • Query:      [6, 8] → 6 tokens, 2 heads × head_size=4 (multi-query)
//    • Key:        [6, 4] → 6 tokens, 1 head × head_size=4
//    • Value:      [6, 6] → 6 tokens, 1 head × head_size=6
//    • Key Cache:   [2, 1, 3, 4] → 2 blocks, 1 head, block_size=3, k_head_size=4
//    • Value Cache: [2, 1, 3, 6] → 2 blocks, 1 head, block_size=3, v_head_size=6
//    • past_lens: {0,3}, subsequence_begins: {0,3,6}
//    • block_indices: {-1,-1}, block_indices_begins: {0,0,0}
//    • Misc: scale=default, sliding_window=2, alibi_slopes={0.25}, max_context_len=4
//    • Rotation: rotated_block_indices={1}, rotation_deltas shape={1,3}, rotation_trig_lut shape={2,4}
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedAttention_advanced,
    PagedAttentionLayerTest,
    ::testing::Combine(
        // 1) Static shapes
        ::testing::Values(ov::test::static_shapes_to_test_representation({
            {6, 8},          // query:       6 tokens, 2 heads × head_size=4
            {6, 4},          // key:         6 tokens, 1 head  × head_size=4
            {6, 6},          // value:       6 tokens, 1 head  × head_size=6
            {2, 1, 3, 4},    // key_cache:   2 blocks, 1 head, block_size=3, head_size=4
            {2, 1, 3, 6}     // value_cache: 2 blocks, 1 head, block_size=3, head_size=6
        })),
        // 2) Integer-vector inputs
        ::testing::Values(ov::test::PagedAttentionIntVectorsStruct{
            /* past_lens                */ {0, 3},
            /* subsequence_begins       */ {0, 3, 6},
            /* block_indices            */ {-1, -1},
            /* block_indices_begins     */ {0,  0,  0}
        }),
        // 3) Miscellaneous inputs
        ::testing::Values(ov::test::PagedAttentionMiscInpStruct{
            /* scale                   */ {},             // default = 1/√6
            /* sliding_window          */ 2,
            /* alibi_slopes            */ {0.25f},
            /* max_context_len         */ 4
        }),
        // 4) Rotation inputs
        ::testing::Values(ov::test::PagedAttentionRotationStruct{
            /* rotated_block_indices   */ {1},
            /* rotation_deltas shape   */ ov::Shape{1, 3},   // 1×3
            /* rotation_trig_lut shape */ ov::Shape{2, 4}    // 2×4
        }),
        // 5) Precision: f32, f16, bf16
        ::testing::ValuesIn(inputPrecisions),
        // 6) Device
        ::testing::Values(ov::test::utils::DEVICE_CPU)
    ),
    PagedAttentionLayerTest::getTestCaseName
);
}  // namespace
