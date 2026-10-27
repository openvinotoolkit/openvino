// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/sparse_pa.hpp"

#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

namespace {

constexpr size_t kNumHeads = 4;

// Steps in the PagedAttention tests' [L, B, H, S] form: consecutive calls of the same B sequences with L new tokens
// each.
std::vector<InputShape> make_steps(size_t batch, size_t head_size, const std::vector<size_t>& lens) {
    std::vector<ov::Shape> targets;
    for (const auto len : lens) {
        targets.push_back({len, batch, kNumHeads, head_size});
    }
    const auto b = static_cast<int64_t>(batch);
    const auto s = static_cast<int64_t>(head_size);
    return {{ov::PartialShape{-1, b, static_cast<int64_t>(kNumHeads), s}, targets}};
}

// Decode steps at the positions 3, 4, 5, ... after a 3-token prefill: the causal boundaries around key_block_size 2 and
// 4 (the position ends a block, starts one, and is the second token of one), where the incomplete tail block of the
// query is empty or not and a block becomes selectable.
const std::vector<size_t> kBoundarySteps = {3, 1, 1, 1, 1, 1, 1, 5, 1};
// A prefill that stops in the middle of a page, a chunked prefill that crosses a page boundary (29 + 8 > 32), decode
// steps on the partially filled pages and another chunk: every call continues the caches the previous ones wrote.
const std::vector<size_t> kChunkedSteps = {29, 8, 1, 1, 32, 1};

std::vector<std::vector<InputShape>> make_shapes(size_t head_size) {
    std::vector<std::vector<InputShape>> shapes;
    for (const size_t batch : {1, 3}) {
        shapes.push_back(make_steps(batch, head_size, kBoundarySteps));
        shapes.push_back(make_steps(batch, head_size, kChunkedSteps));
    }
    return shapes;
}

// Mixed batches need at least three sequences to get a prefill, a decode and a shorter chunk in one call; four also
// repeat the prefill pattern.
std::vector<std::vector<InputShape>> make_mixed_shapes(size_t head_size) {
    std::vector<std::vector<InputShape>> shapes;
    for (const size_t batch : {3, 4}) {
        shapes.push_back(make_steps(batch, head_size, kBoundarySteps));
        shapes.push_back(make_steps(batch, head_size, kChunkedSteps));
    }
    return shapes;
}

std::vector<SparsePAConfig> as_mixed(std::vector<SparsePAConfig> configs) {
    for (auto& cfg : configs) {
        cfg.mixed_lengths = true;
    }
    return configs;
}

// The same configs on a single thread, which makes the kernel process whole query-head packs in every work item.
std::vector<SparsePAConfig> single_thread(std::vector<SparsePAConfig> configs) {
    for (auto& cfg : configs) {
        cfg.num_threads = 1;
    }
    return configs;
}

SparsePAConfig plain_config(size_t num_selection_heads,
                            size_t key_block_size,
                            size_t page_size,
                            size_t max_selected_blocks,
                            SparsePASelection selection) {
    SparsePAConfig cfg;
    cfg.num_selection_heads = num_selection_heads;
    cfg.key_block_size = key_block_size;
    cfg.page_size = page_size;
    cfg.max_selected_blocks = max_selected_blocks;
    cfg.selection = selection;
    return cfg;
}

SparsePAConfig quantized_config(ov::element::Type cache_precision,
                                bool key_bychannel,
                                size_t group_size,
                                size_t num_selection_heads,
                                SparsePASelection selection) {
    SparsePAConfig cfg;
    cfg.num_selection_heads = num_selection_heads;
    cfg.key_block_size = 4;
    cfg.page_size = 32;  // the page size ConvertPagedAttnInputs configures
    cfg.max_selected_blocks = 8;
    cfg.cache_precision = cache_precision;
    cfg.key_bychannel = key_bychannel;
    cfg.group_size = group_size;
    cfg.selection = selection;
    return cfg;
}

std::vector<SparsePAConfig> plain_configs() {
    std::vector<SparsePAConfig> configs;
    for (const auto selection : {SparsePASelection::kRecent, SparsePASelection::kRandom, SparsePASelection::kEmpty}) {
        // A selection head shared by every query head, one per KV head, one per query head. A page of 4 or 8 tokens
        // holds 2 key blocks, so selected blocks regularly straddle pages.
        configs.push_back(plain_config(1, 2, 4, 4, selection));
        configs.push_back(plain_config(2, 2, 8, 4, selection));
        configs.push_back(plain_config(kNumHeads, 2, 8, 4, selection));
    }
    return configs;
}

// AMX QK^T fast path: bf16 end to end, head size a multiple of 32, key_block_size 4 (the Qwen production shape, smaller
// than the JIT kernel's 16-row granularity, so block merging is exercised) with a page that fits 4 merged blocks and
// one that only fits a single 16-row unit. Falls back to the scalar kernel transparently on non-AMX hardware.
std::vector<SparsePAConfig> amx_configs() {
    std::vector<SparsePAConfig> configs;
    for (const auto selection : {SparsePASelection::kRecent, SparsePASelection::kRandom}) {
        configs.push_back(plain_config(1, 4, 16, 4, selection));
        configs.push_back(plain_config(1, 4, 64, 4, selection));
        configs.push_back(plain_config(2, 4, 64, 16, selection));
    }
    return configs;
}

// Quantized (u8/u4) KV caches, as produced by ConvertPagedAttnInputs: by-token or by-channel keys, by-token values, one
// group per row or several.
std::vector<SparsePAConfig> quantized_configs() {
    std::vector<SparsePAConfig> configs;
    for (const auto precision : {ov::element::u8, ov::element::u4}) {
        for (const bool key_bychannel : {false, true}) {
            for (const size_t group_size : {0, 16}) {
                for (const size_t num_selection_heads : {1, 2}) {
                    for (const auto selection : {SparsePASelection::kRecent, SparsePASelection::kRandom}) {
                        configs.push_back(
                            quantized_config(precision, key_bychannel, group_size, num_selection_heads, selection));
                    }
                }
            }
        }
    }
    return configs;
}

// Qwen-like shapes for the DISABLED_Perf benchmark: 24 query heads over 2 K heads, head size 256, a selection of 128
// blocks of 4 tokens, one prefill call followed by decode calls, and the same as a mixed batch.
std::vector<std::vector<InputShape>> perf_shapes() {
    const auto steps = [](size_t batch) {
        std::vector<ov::Shape> targets = {{1024, batch, 24, 256}, {1, batch, 24, 256}};
        return std::vector<InputShape>{{ov::PartialShape{-1, static_cast<int64_t>(batch), 24, 256}, targets}};
    };
    return {steps(1), steps(3)};
}

std::vector<SparsePAConfig> perf_configs() {
    std::vector<SparsePAConfig> configs;
    for (const auto precision : {ov::element::dynamic, ov::element::u8, ov::element::u4}) {
        for (const bool mixed : {false, true}) {
            SparsePAConfig cfg;
            cfg.num_kv_heads = 2;
            cfg.num_selection_heads = 1;
            cfg.v_head_size = 256;
            cfg.key_block_size = 4;
            cfg.page_size = 32;
            cfg.max_selected_blocks = 128;
            cfg.cache_precision = precision;
            cfg.key_bychannel = precision != ov::element::dynamic;
            cfg.mixed_lengths = mixed;
            configs.push_back(cfg);
        }
    }
    return configs;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(perf_SparsePA_CPU,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16),
                                          testing::ValuesIn(perf_shapes()),
                                          testing::ValuesIn(perf_configs())),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_Plain,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::ValuesIn(make_shapes(16)),
                                          testing::ValuesIn(plain_configs())),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_AMX,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16),
                                          testing::ValuesIn(make_shapes(32)),
                                          testing::ValuesIn(amx_configs())),
                         SparsePALayerCPUTest::getTestCaseName);

// Decode, prefill and chunked-prefill sequences in the same call (sequence s % 3 picks the kind), with different past
// lengths, for plain and quantized caches and every selection-head count.
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_MixedBatchPlain,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(16)),
                                          testing::ValuesIn(as_mixed(plain_configs()))),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_MixedBatchAMX,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(32)),
                                          testing::ValuesIn(as_mixed(amx_configs()))),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_MixedBatchQuantizedCache,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(32)),
                                          testing::ValuesIn(as_mixed(quantized_configs()))),
                         SparsePALayerCPUTest::getTestCaseName);

// Packed query heads: one thread, so every work item holds a whole pack (see SparsePAConfig::num_threads).
INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_PackedPlain,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(16)),
                                          testing::ValuesIn(single_thread(as_mixed(plain_configs())))),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_PackedAMX,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(32)),
                                          testing::ValuesIn(single_thread(as_mixed(amx_configs())))),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_PackedQuantizedCache,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::ValuesIn(make_mixed_shapes(32)),
                                          testing::ValuesIn(single_thread(as_mixed(quantized_configs())))),
                         SparsePALayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparsePA_CPU_QuantizedCache,
                         SparsePALayerCPUTest,
                         testing::Combine(testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::ValuesIn(make_shapes(32)),
                                          testing::ValuesIn(quantized_configs())),
                         SparsePALayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
