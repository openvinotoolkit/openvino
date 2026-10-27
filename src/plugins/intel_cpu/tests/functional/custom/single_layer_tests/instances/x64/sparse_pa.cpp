// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_pa.hpp"

#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

namespace {

constexpr size_t kNumHeads = 4;

// Decode steps at the positions 3, 4, 5, ... after a 3-token prefill: the causal boundaries around key_block_size 2 and
// 4 (the position ends a block, starts one, and is the second token of one), where the incomplete tail block of the
// query is empty or not and a block becomes selectable.
const std::vector<size_t> kBoundarySteps = {3, 1, 1, 1, 1, 1, 1, 5, 1};
// A prefill that stops in the middle of a page, a chunked prefill that crosses a page boundary (29 + 8 > 32), decode
// steps on the partially filled pages and another chunk: every call continues the caches the previous ones wrote.
const std::vector<size_t> kChunkedSteps = {29, 8, 1, 1, 32, 1};

enum class Batch {
    kSingle,  // one sequence
    kMixed,   // three sequences: a prefill, a decode and a shorter chunk in the same call (see SparsePAConfig)
};

// One test: the steps follow the PagedAttention tests' [L, B, H, S] form (consecutive calls of the same B sequences).
SparsePALayerTestParams make_case(ov::element::Type data_precision, size_t head_size, Batch batch, SparsePAConfig cfg) {
    const bool mixed = batch == Batch::kMixed;
    const size_t sequences = mixed ? 3 : 1;
    std::vector<ov::Shape> targets;
    for (const auto len : mixed ? kChunkedSteps : kBoundarySteps) {
        targets.push_back({len, sequences, kNumHeads, head_size});
    }
    cfg.mixed_lengths = mixed;
    const std::vector<InputShape> steps = {{ov::PartialShape{-1,
                                                             static_cast<int64_t>(sequences),
                                                             static_cast<int64_t>(kNumHeads),
                                                             static_cast<int64_t>(head_size)},
                                            targets}};
    return {data_precision, steps, cfg};
}

SparsePAConfig plain_config(size_t sel_head_num,
                            size_t key_block_size,
                            size_t page_size,
                            size_t max_selected_blocks,
                            SparsePASelection selection,
                            size_t v_head_size = 16,
                            size_t num_threads = 0) {
    SparsePAConfig cfg;
    cfg.num_selection_heads = sel_head_num;
    cfg.v_head_size = v_head_size;
    cfg.key_block_size = key_block_size;
    cfg.page_size = page_size;
    cfg.max_selected_blocks = max_selected_blocks;
    cfg.selection = selection;
    cfg.num_threads = num_threads;
    return cfg;
}

// Quantized caches as ConvertPagedAttnInputs produces them: 32-token pages, key_block_size 4.
SparsePAConfig quantized_config(ov::element::Type cache_precision,
                                bool key_bychannel,
                                size_t group_size,
                                size_t sel_head_num,
                                SparsePASelection selection,
                                size_t v_head_size = 16,
                                size_t max_selected_blocks = 8,
                                size_t num_threads = 0) {
    auto cfg = plain_config(sel_head_num, 4, 32, max_selected_blocks, selection, v_head_size, num_threads);
    cfg.cache_precision = cache_precision;
    cfg.key_bychannel = key_bychannel;
    cfg.group_size = group_size;
    return cfg;
}

constexpr auto kRecent = SparsePASelection::kRecent;
constexpr auto kRandom = SparsePASelection::kRandom;
constexpr auto kEmpty = SparsePASelection::kEmpty;

// Plain caches on the vector kernels: every data precision, selection head counts of 1 / K heads / query heads, tiny
// pages (selected blocks straddle pages), every selection mode, and a single thread, which makes every work item a
// whole query-head pack (the kernel packs only when there is enough work for every thread).
std::vector<SparsePALayerTestParams> plain_cases() {
    const auto f32 = ov::element::f32;
    const auto f16 = ov::element::f16;
    const auto bf16 = ov::element::bf16;
    return {make_case(f32, 16, Batch::kSingle, plain_config(1, 2, 4, 4, kRecent)),
            make_case(f16, 16, Batch::kSingle, plain_config(2, 2, 8, 4, kRandom)),
            make_case(bf16, 16, Batch::kMixed, plain_config(kNumHeads, 2, 8, 4, kRandom)),
            make_case(f32, 16, Batch::kMixed, plain_config(1, 2, 4, 4, kEmpty)),
            make_case(bf16, 16, Batch::kMixed, plain_config(2, 2, 8, 4, kRecent, 16, 1)),
            make_case(f16, 16, Batch::kMixed, plain_config(1, 2, 4, 4, kRandom, 16, 1))};
}

// AMX QK^T path, bf16 end to end (AMX f16 needs amx_fp16): key_block_size 4 (smaller than the JIT kernel's 16-row
// granularity, so adjacent blocks are merged) with pages that fit several merged blocks or only one unit. Head sizes 32
// and 128 fit the kernel's six B tiles; 256 (the Qwen head size, 4 + 4 tiles) and 512 (6 + 5 + 5, an uneven split) take
// its K-chunk loop. v_head_size differs from the key head size to catch any coupling of the two.
std::vector<SparsePALayerTestParams> amx_cases() {
    const auto bf16 = ov::element::bf16;
    return {make_case(bf16, 32, Batch::kMixed, plain_config(1, 4, 16, 4, kRecent)),
            make_case(bf16, 32, Batch::kMixed, plain_config(1, 4, 64, 16, kRandom)),
            make_case(bf16, 32, Batch::kMixed, plain_config(2, 4, 64, 16, kRecent, 16, 1)),
            make_case(bf16, 128, Batch::kMixed, plain_config(1, 4, 32, 8, kRandom, 64)),
            make_case(bf16, 256, Batch::kMixed, plain_config(1, 4, 16, 8, kRecent, 256)),
            make_case(bf16, 256, Batch::kMixed, plain_config(1, 4, 64, 8, kRandom, 128)),
            make_case(bf16, 256, Batch::kMixed, plain_config(2, 4, 32, 8, kRandom, 256, 1)),
            make_case(bf16, 512, Batch::kMixed, plain_config(1, 4, 32, 8, kRandom, 256))};
}

// Quantized (u8 / u4) caches: by-token and by-channel keys, one group per row or several, one or several selection
// heads, and a single thread (query-head packs). Quantized chunks are dequantized to the AMX type, so with bf16 data
// and a head size that fits the AMX kernel the prefill steps run on it (quantized decode uses the fused
// direct-quantized kernels).
std::vector<SparsePALayerTestParams> quantized_cases() {
    const auto f32 = ov::element::f32;
    const auto bf16 = ov::element::bf16;
    const auto u8 = ov::element::u8;
    const auto u4 = ov::element::u4;
    return {make_case(f32, 32, Batch::kMixed, quantized_config(u8, false, 0, 1, kRecent)),
            make_case(f32, 32, Batch::kMixed, quantized_config(u4, true, 0, 2, kRandom)),
            make_case(bf16, 32, Batch::kMixed, quantized_config(u8, true, 16, 1, kRandom)),
            make_case(bf16, 32, Batch::kMixed, quantized_config(u4, false, 16, 1, kRecent)),
            make_case(bf16, 32, Batch::kMixed, quantized_config(u8, true, 0, 2, kRandom, 16, 8, 1)),
            make_case(bf16, 32, Batch::kMixed, quantized_config(u4, false, 16, 1, kRecent, 16, 8, 1)),
            // 16 selected blocks span two 32-token pages, so four adjacent blocks inside one page (an AMX chunk) occur.
            make_case(bf16, 256, Batch::kMixed, quantized_config(u8, true, 0, 1, kRecent, 128, 16)),
            make_case(bf16, 256, Batch::kMixed, quantized_config(u4, false, 16, 1, kRandom, 256, 16, 1))};
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_Plain,
                         SparsePALayerCPUTest,
                         testing::ValuesIn(plain_cases()),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_AMX,
                         SparsePALayerCPUTest,
                         testing::ValuesIn(amx_cases()),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_QuantizedCache,
                         SparsePALayerCPUTest,
                         testing::ValuesIn(quantized_cases()),
                         SparsePALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
