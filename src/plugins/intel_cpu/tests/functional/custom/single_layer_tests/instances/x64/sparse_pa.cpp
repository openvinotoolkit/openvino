// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_pa.hpp"

#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

namespace {
constexpr size_t kU = 1;  // readability helper for the size_t Values() lists below
}  // namespace

// num_selection_heads == 1: a single selection head shared by every query head. page_size 4 is a
// multiple of key_block_size 2, so a selected block never straddles two physical pages.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_Hsel1,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(kU),
                                          testing::Values(true, false),
                                          testing::Values(kU, static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(true)),
                         SparsePALayerCPUTest::getTestCaseName);

// num_selection_heads == num_kv_heads: exercises the sh = h / (Hq/num_selection_heads) grouping path.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_HselEqualsHkv,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(8)),
                                          testing::Values(true)),
                         SparsePALayerCPUTest::getTestCaseName);

// The optional QB_START input omitted (falls back to subsequence_begins).
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_NoQbStart,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(kU),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(false)),
                         SparsePALayerCPUTest::getTestCaseName);

// AMX QK^T fast path: bf16 end-to-end, k_head_size % 32 == 0, key_block_size == 4 (the Qwen4
// production shape, smaller than the JIT kernel's 16-row granularity, so block merging is exercised)
// with a page that fits 4 merged blocks and one that only fits a single 16-row unit. Falls back to
// the scalar kernel transparently on non-AMX hardware.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_AMX,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::bf16),
                                          testing::Values(kU),
                                          testing::Values(true, false),
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(32)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(16), static_cast<size_t>(64)),
                                          testing::Values(true)),
                         SparsePALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
