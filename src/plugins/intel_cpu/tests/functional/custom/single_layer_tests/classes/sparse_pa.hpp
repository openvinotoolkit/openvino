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

// Everything about the op that is not carried by the step shapes.
struct SparsePAConfig {
    size_t num_kv_heads = 2;
    size_t num_selection_heads = 1;  // 1, num_kv_heads, or the number of query heads
    size_t v_head_size = 16;
    size_t key_block_size = 2;
    size_t page_size = 4;  // tokens per physical cache block; a multiple of key_block_size
    size_t max_selected_blocks = 4;
    // u8/u4: quantized KV caches, laid out as PagedAttention does. Anything else: plain caches at the data precision.
    ov::element::Type cache_precision = ov::element::dynamic;
    bool key_bychannel = false;  // quantized key cache by channel, else by token (the value cache is always by token)
    size_t group_size = 0;       // by-token quantization group, 0: the whole row
    SparsePASelection selection = SparsePASelection::kRecent;
};

// Data precision (query, key, value, scale, sinks, plain caches and the inference precision), the steps and the config.
// The steps follow the PagedAttention tests: one InputShape whose target shapes are consecutive calls of the same
// sequences, each [L, B, H, S] = L new tokens in each of B sequences, H query heads, S key head size. The caches
// persist across the calls, so every call continues the history written by the previous ones: L > 1 is a (chunked)
// prefill, L == 1 a decode step.
using SparsePALayerTestParams = std::tuple<ov::element::Type, std::vector<InputShape>, SparsePAConfig>;

class SparsePALayerCPUTest : public testing::WithParamInterface<SparsePALayerTestParams>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    [[nodiscard]] bool quantized() const {
        return m_cfg.cache_precision == ov::element::u8 || m_cfg.cache_precision == ov::element::u4;
    }
    // Final key/value cache shapes: quantization parameters live inside the cache tensor (see SparsePAQuant).
    [[nodiscard]] ov::Shape cache_shape(bool is_key) const;

    ov::element::Type m_data_precision = ov::element::f32;
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
    size_t m_total_len = 0;  // tokens per sequence written by the steps generated so far
    size_t m_cur_past = 0;   // tokens per sequence before the current step
    size_t m_cur_len = 0;    // new tokens per sequence in the current step
    ov::Tensor m_key_cache;
    ov::Tensor m_value_cache;
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;

    // Oracle state: effective (dequantized) key/value rows of every sequence and kv head, [sequence][kv] -> [pos, S].
    std::vector<std::vector<std::vector<float>>> m_eff_key;
    std::vector<std::vector<std::vector<float>>> m_eff_value;
};

}  // namespace test
}  // namespace ov
