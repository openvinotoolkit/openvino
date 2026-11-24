// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// How the synthetic indexer picks the complete blocks every new token attends to.
enum class SparsePASelection {
    kRecent,  // the most recent complete blocks, up to max_selected_blocks
    kRandom,  // a random number (zero included) of random distinct complete blocks, in ascending order
    kEmpty,   // none: only the causal tail is attended
};

// An invalid input the op must reject; such caches are declared with dynamic page / head dims, as in a real model.
enum class SparsePAInvalid {
    kNone,
    kValueCachePageSize,      // the value cache has fewer rows per page than the key cache
    kValueCacheHeadSize,      // the value cache rows are narrower than v_head_size
    kKvHeadCount,             // both caches have one KV head too few
    kBlockIdOutOfRange,       // a block id equal to the number of cache pages
    kNegativeBlockId,         // a block id of -1
    kSubsequenceBeginsShort,  // subsequence_begins ends before the last token
};

// Everything about the op that is not carried by the step shapes.
struct SparsePAConfig {
    size_t num_kv_heads = 2;
    size_t num_selection_heads = 1;  // 1, num_kv_heads, or the number of query heads
    size_t v_head_size = 16;
    size_t key_block_size = 2;
    size_t page_size = 4;  // tokens per physical cache block; a multiple of key_block_size
    size_t max_selected_blocks = 4;
    // u8/u4: quantized KV caches, laid out as PagedAttention does. f16 with f32 data: f16 plain caches. dynamic: plain
    // caches at the data precision.
    ov::element::Type cache_precision = ov::element::dynamic;
    bool key_bychannel = false;  // quantized key cache by channel, else by token (the value cache is always by token)
    size_t group_size = 0;       // by-token quantization group, 0: the whole row
    SparsePASelection selection = SparsePASelection::kRecent;
    // the selected blocks of a row in random order instead of ascending: the order does not matter to SparseAttention
    bool shuffle_blocks = false;
    // false: every sequence has the step's L tokens. true: a mixed batch, sequence s gets L, 1 or (L + 1) / 2 tokens
    // for s % 3 == 0, 1, 2.
    bool mixed_lengths = false;
    SparsePAInvalid invalid = SparsePAInvalid::kNone;  // plain caches only
};

// steps: one InputShape, targets [L, B, H, S] are consecutive calls on B sequences (L > 1 prefill, L == 1 decode).
// inference_precision: dynamic = data precision; another one adds a Convert behind the output.
using SparsePALayerTestParams = std::tuple<ov::element::Type,        // data_precision
                                           std::vector<InputShape>,  // steps
                                           SparsePAConfig,           // config
                                           ov::element::Type>;       // inference_precision

class SparsePALayerCPUTest : public testing::WithParamInterface<SparsePALayerTestParams>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void check_runtime_precision() const;

private:
    [[nodiscard]] bool quantized() const {
        return m_cfg.cache_precision == ov::element::u8 || m_cfg.cache_precision == ov::element::u4;
    }
    // Element type of the cache tensors handed to the op.
    [[nodiscard]] ov::element::Type cache_type() const {
        return m_cfg.cache_precision == ov::element::dynamic ? m_data_precision : m_cfg.cache_precision;
    }
    // A value as the op reads it back from a plain cache of cache_type().
    [[nodiscard]] float through_cache(float value) const;
    // New tokens of `sequence` in a step whose shape says `step_len`.
    [[nodiscard]] size_t sequence_len(size_t step_len, size_t sequence) const;
    // Final key/value cache shapes: quantization parameters live inside the cache tensor (see CacheSpec).
    [[nodiscard]] ov::Shape cache_shape(bool is_key) const;

    ov::element::Type m_data_precision = ov::element::f32;
    ov::element::Type m_inference_precision = ov::element::f32;  // the hint the model runs with
    SparsePAConfig m_cfg;
    // Step shapes [L, B, H, S]
    std::vector<ov::Shape> m_steps;
    size_t m_num_sequences = 1;
    size_t m_num_heads = 4;
    size_t m_k_head_size = 16;
    size_t m_pages_per_sequence = 1;
    float m_scale = 0.25F;

    // Step bookkeeping. The caches are created by the first step and reused by the next ones.
    size_t m_step = 0;
    std::vector<size_t> m_total_len;  // per sequence: tokens written by the steps generated so far
    std::vector<size_t> m_cur_past;   // per sequence: tokens before the current step
    std::vector<size_t> m_cur_len;    // per sequence: new tokens in the current step
    ov::Tensor m_key_cache;
    ov::Tensor m_value_cache;
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;

    // Oracle state: effective (dequantized) key/value rows of every sequence and kv head, [sequence][kv] -> [pos, S].
    std::vector<std::vector<std::vector<float>>> m_eff_key;
    std::vector<std::vector<std::vector<float>>> m_eff_value;
};

}  // namespace test
}  // namespace ov
