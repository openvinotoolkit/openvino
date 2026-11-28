// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_sdpa.hpp"

#include <vector>

#include "utils/cpu_test_utils.hpp"

namespace ov::test {

namespace {

const std::vector<ov::element::Type> all_precisions{ov::element::f32, ov::element::f16, ov::element::bf16};

// The small shape of the smoke tests (L == S == 40, up to 10 selected blocks, adjacent blocks) is the default of the
// arguments, the other cases set what they exercise.
SparseSDPALayerTestParams make_case(ov::element::Type precision,
                                    size_t h_sel,
                                    bool causal,
                                    size_t batch,
                                    size_t k_head_size,
                                    size_t sel_block_size,
                                    std::vector<size_t> key_length = {},
                                    size_t seq_len = 40,
                                    size_t max_blocks = 10,
                                    size_t block_step = 1,
                                    bool shuffle = false,
                                    size_t num_q_heads = 4,
                                    size_t num_kv_heads = 2,
                                    size_t v_head_size = 16) {
    return {batch,
            num_q_heads,
            num_kv_heads,
            k_head_size,
            v_head_size,
            seq_len,
            h_sel,
            sel_block_size,
            max_blocks,
            block_step,
            causal,
            shuffle,
            std::move(key_length),
            precision,
            ov::test::utils::DEVICE_CPU};
}

// h_sel == 1: a single selection head shared by every query head. h_sel == Hkv: one selection head per kv head,
// exercises the sh = h / (Hq/h_sel) grouping path.
std::vector<SparseSDPALayerTestParams> selection_head_cases() {
    std::vector<SparseSDPALayerTestParams> cases;
    for (const size_t h_sel : {1, 2}) {
        for (const auto precision : all_precisions) {
            for (const bool causal : {true, false}) {
                for (const size_t batch : {1, 2}) {
                    cases.push_back(make_case(precision, h_sel, causal, batch, 16, 2));
                }
            }
        }
    }
    return cases;
}

// Other head layouts: a group of 4 query heads per KV head, one query head per KV head, and a value head size that
// differs from the key one.
std::vector<SparseSDPALayerTestParams> head_layout_cases() {
    std::vector<SparseSDPALayerTestParams> cases;
    for (const auto precision : {ov::element::f32, ov::element::bf16}) {
        for (const size_t h_sel : {1, 2}) {
            const auto layout = [&](size_t num_q_heads, size_t num_kv_heads, size_t v_head_size) {
                return make_case(precision,
                                 h_sel,
                                 true,
                                 2,
                                 16,
                                 2,
                                 {},
                                 40,
                                 10,
                                 1,
                                 false,
                                 num_q_heads,
                                 num_kv_heads,
                                 v_head_size);
            };
            cases.push_back(layout(8, 2, 16));
            cases.push_back(layout(2, 2, 16));
            cases.push_back(layout(4, 2, 32));
        }
    }
    return cases;
}

// AMX paths: bf16, k_head_size % 32 == 0, sel_block_size 4; 256 takes the K-chunk loop. Other hardware uses SIMD
// kernels.
std::vector<SparseSDPALayerTestParams> amx_cases() {
    std::vector<SparseSDPALayerTestParams> cases;
    for (const bool causal : {true, false}) {
        for (const size_t k_head_size : {32, 256}) {
            cases.push_back(make_case(ov::element::bf16, 1, causal, 1, k_head_size, 4));
        }
    }
    return cases;
}

// Left-padded batches with KEY_LENGTH (L == 40): no padding, padded off a block boundary (37), on one (32), almost all
// padding (4). The last set has an AMX-eligible shape.
std::vector<SparseSDPALayerTestParams> padded_cases() {
    std::vector<SparseSDPALayerTestParams> cases;
    for (const auto precision : {ov::element::f32, ov::element::bf16}) {
        for (const bool causal : {true, false}) {
            cases.push_back(make_case(precision, 1, causal, 4, 16, 2, {40, 37, 32, 4}));
        }
    }
    for (const bool causal : {true, false}) {
        cases.push_back(make_case(ov::element::bf16, 1, causal, 4, 32, 4, {40, 37, 32, 4}));
    }
    return cases;
}

// AMX prefill over packs of query heads (bf16, block size 4, causal): non-adjacent selection, a selection head per
// query head, and prefill lengths around 32 and 128 rows.
std::vector<SparseSDPALayerTestParams> packed_cases() {
    const auto packed = [](size_t h_sel, size_t k_head_size, size_t seq_len, size_t max_blocks, size_t block_step) {
        return make_case(ov::element::bf16, h_sel, true, 1, k_head_size, 4, {}, seq_len, max_blocks, block_step);
    };
    return {packed(/*h_sel=*/1, /*k_head_size=*/256, /*seq_len=*/40, /*max_blocks=*/10, /*block_step=*/2),
            packed(/*h_sel=*/4, /*k_head_size=*/32, /*seq_len=*/40, /*max_blocks=*/10, /*block_step=*/1),
            packed(/*h_sel=*/1, /*k_head_size=*/32, /*seq_len=*/31, /*max_blocks=*/128, /*block_step=*/1),
            packed(/*h_sel=*/1, /*k_head_size=*/32, /*seq_len=*/32, /*max_blocks=*/128, /*block_step=*/1),
            packed(/*h_sel=*/1, /*k_head_size=*/32, /*seq_len=*/254, /*max_blocks=*/128, /*block_step=*/1),
            packed(/*h_sel=*/1, /*k_head_size=*/256, /*seq_len=*/256, /*max_blocks=*/128, /*block_step=*/1)};
}

// The selected blocks in random order instead of ascending (SparseAttention does not depend on the order, no adjacent
// runs remain): bf16 with the AMX head sizes, the non-AMX precisions, the non-adjacent step and padded batches.
std::vector<SparseSDPALayerTestParams> shuffled_cases() {
    std::vector<SparseSDPALayerTestParams> cases;
    for (const size_t h_sel : {1, 2}) {
        for (const bool causal : {true, false}) {
            for (const size_t batch : {1, 2}) {
                for (const size_t k_head_size : {32, 256}) {
                    cases.push_back(
                        make_case(ov::element::bf16, h_sel, causal, batch, k_head_size, 4, {}, 40, 10, 1, true));
                    cases.push_back(
                        make_case(ov::element::bf16, h_sel, causal, batch, k_head_size, 4, {}, 40, 10, 2, true));
                    cases.push_back(
                        make_case(ov::element::bf16, h_sel, causal, batch, k_head_size, 4, {}, 96, 20, 1, true));
                }
            }
        }
    }
    for (const auto precision : {ov::element::f32, ov::element::f16}) {
        for (const bool causal : {true, false}) {
            for (const size_t batch : {1, 2}) {
                cases.push_back(make_case(precision, 1, causal, batch, 32, 4, {}, 96, 20, 1, true));
            }
        }
    }
    for (const bool causal : {true, false}) {
        cases.push_back(make_case(ov::element::bf16, 1, causal, 2, 32, 4, {96, 70}, 96, 20, 1, true));
    }
    return cases;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_SelectionHeads,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(selection_head_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_HeadLayouts,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(head_layout_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_AMX,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(amx_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_Padded,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(padded_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_AMX_Packed,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(packed_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseSDPA_CPU_Shuffled,
                         SparseSDPALayerCPUTest,
                         testing::ValuesIn(shuffled_cases()),
                         SparseSDPALayerCPUTest::getTestCaseName);

}  // namespace ov::test
