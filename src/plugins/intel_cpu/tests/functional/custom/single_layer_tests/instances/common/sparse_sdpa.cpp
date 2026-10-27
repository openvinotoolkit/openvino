// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_sdpa.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

// h_sel == 1: a single selection head shared by every query head.
INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_Hsel1,
                         SparseSDPALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(std::vector<size_t>{}),
                                          testing::Values(SparseSDPAShape{})),
                         SparseSDPALayerCPUTest::getTestCaseName);

// h_sel == Hkv: one selection head per kv head, exercises the sh = h / (Hq/h_sel) grouping path.
INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_HselEqualsHkv,
                         SparseSDPALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(std::vector<size_t>{}),
                                          testing::Values(SparseSDPAShape{})),
                         SparseSDPALayerCPUTest::getTestCaseName);

// AMX QK^T fast path: bf16 end-to-end, k_head_size % 32 == 0, sel_block_size == 4 -- the real
// Qwen4 production shape (smaller than the JIT kernel's native 16-row granularity, so this also
// exercises the m_amx_unit_blocks block-merging path). Falls back to the scalar kernel
// transparently on non-AMX hardware -- same pass/fail bar either way.
INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_AMX,
                         SparseSDPALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::bf16),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(1)),
                                          // 256 (the Qwen head size) is above the 192 that fit the six B tiles of the
                                          // AMX kernel and takes its K-chunk loop.
                                          testing::Values(static_cast<size_t>(32), static_cast<size_t>(256)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(std::vector<size_t>{}),
                                          testing::Values(SparseSDPAShape{})),
                         SparseSDPALayerCPUTest::getTestCaseName);

// Left-padded batches: KEY_LENGTH input present, one entry per batch item (kCacheLen == kL == 40,
// see classes/sparse_sdpa.cpp). Covers a batch item with no padding at all (40), one padded short
// of a sel_block_size boundary (37 -- exercises the pad-not-block-aligned causal-tail path), one
// padded to exactly a sel_block_size boundary (32), and one so short that some leading queries have
// no valid history whatsoever (4 -- query rows l < pad must produce all-zero output).
INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_Padded,
                         SparseSDPALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(std::vector<size_t>{40, 37, 32, 4}),
                                          testing::Values(SparseSDPAShape{})),
                         SparseSDPALayerCPUTest::getTestCaseName);

// Same left-padded batches as smoke_SparseSDPA_CPU_Padded, but with the AMX-eligible shape
// (k_head_size % 32 == 0, see createPrimitive()'s AMX gate in sparse_sdpa.cpp) so the padded
// code path through run_amx_qk_chunks()/decompose_runs() (not just the scalar fallback) gets
// exercised on AMX-capable hardware. Falls back to the scalar kernel transparently otherwise --
// same pass/fail bar either way, same as smoke_SparseSDPA_CPU_AMX.
INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_AMX_Padded,
                         SparseSDPALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::bf16),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(32)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(std::vector<size_t>{40, 37, 32, 4}),
                                          testing::Values(SparseSDPAShape{})),
                         SparseSDPALayerCPUTest::getTestCaseName);

namespace {

SparseSDPALayerTestParams packed_case(size_t h_sel,
                                      size_t batch,
                                      size_t k_head_size,
                                      std::vector<size_t> key_length,
                                      SparseSDPAShape shape) {
    return {ov::test::utils::DEVICE_CPU,
            ov::element::bf16,
            h_sel,
            true,
            batch,
            k_head_size,
            4,
            std::move(key_length),
            shape};
}

// AMX prefill over packs of query heads: bf16, sel_block_size 4, causal. The shapes are {seq_len, max_blocks,
// block_step}. The suites above already run the prefill over the 2 query heads of a K head as brgemm products (the test
// derives the worker threads from the shape, see SetUp); these add
//  - a selection of every second block (no adjacent runs) and a selection head per query head (no pack: the per-head
//  AMX
//    path);
//  - prefill lengths around the 32 rows of a brgemm chunk (31, 32) and the 128 rows of a tile (254, 256 tokens: with
//    adjacent blocks the selection of the token at position p has p + 1 rows, so every row count up to the length
//    occurs, the tails of every length, a tile that is exactly full and a partial last tile).
std::vector<SparseSDPALayerTestParams> packed_cases() {
    return {packed_case(1, 1, 256, {}, {40, 10, 2}),
            packed_case(4, 1, 32, {}, {40, 10, 1}),
            packed_case(1, 1, 32, {}, {31, 128, 1}),
            packed_case(1, 1, 32, {}, {32, 128, 1}),
            packed_case(1, 1, 32, {}, {254, 128, 1}),
            packed_case(1, 1, 256, {}, {256, 128, 1})};
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_AMX_Packed,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(packed_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
