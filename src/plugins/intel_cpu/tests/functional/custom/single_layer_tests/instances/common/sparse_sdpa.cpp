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
                                          testing::Values(std::vector<size_t>{})),
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
                                          testing::Values(std::vector<size_t>{})),
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
                                          testing::Values(std::vector<size_t>{})),
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
                                          testing::Values(std::vector<size_t>{40, 37, 32, 4})),
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
                                          testing::Values(std::vector<size_t>{40, 37, 32, 4})),
                         SparseSDPALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
