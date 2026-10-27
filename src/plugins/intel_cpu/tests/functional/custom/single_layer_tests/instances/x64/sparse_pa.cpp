// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_pa.hpp"

#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

namespace {
constexpr size_t kU = 1;  // readability helper for the size_t Values() lists below
constexpr size_t kNoGroup = 0;  // by-token quantization with one group per row
constexpr size_t kDecode = 1;   // one new token per sequence: the decode path
constexpr size_t kPrefill = 8;  // several new tokens per sequence: the prefill path
}  // namespace

// num_selection_heads == 1: a single selection head shared by every query head. page_size 4 is a
// multiple of key_block_size 2, so a selected block never straddles two physical pages.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_Hsel1,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(kU),
                                          testing::Values(kU, static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(ov::element::dynamic),
                                          testing::Values(false),
                                          testing::Values(kNoGroup),
                                          testing::Values(kPrefill)),
                         SparsePALayerCPUTest::getTestCaseName);

// num_selection_heads == num_kv_heads: exercises the sh = h / (Hq/num_selection_heads) grouping path.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_HselEqualsHkv,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(16)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(8)),
                                          testing::Values(ov::element::dynamic),
                                          testing::Values(false),
                                          testing::Values(kNoGroup),
                                          testing::Values(kPrefill)),
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
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(32)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(16), static_cast<size_t>(64)),
                                          testing::Values(ov::element::dynamic),
                                          testing::Values(false),
                                          testing::Values(kNoGroup),
                                          testing::Values(kPrefill)),
                         SparsePALayerCPUTest::getTestCaseName);

// Quantized (u8/u4) KV caches, as produced by ConvertPagedAttnInputs: by-token or by-channel keys, by-token values,
// with one group per row or several, for the decode (1 new token) and prefill (several new tokens) paths. past_len
// values that are not a multiple of the page size make by-channel keys requantize a partially filled leading page.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_QuantizedCache,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(kU, static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(3)),
                                          testing::Values(static_cast<size_t>(32)),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(32)),
                                          testing::Values(ov::element::u8, ov::element::u4),
                                          testing::Values(false, true),
                                          testing::Values(kNoGroup, static_cast<size_t>(16)),
                                          testing::Values(kDecode, kPrefill)),
                         SparsePALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
