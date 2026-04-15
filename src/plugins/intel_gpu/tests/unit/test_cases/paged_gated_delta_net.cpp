// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_gated_delta_net.hpp>
#include <numeric>
#include <vector>

#include "paged_gated_delta_net_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct paged_gated_delta_net_test_params {
    std::vector<int32_t> subsequence_tokens;  // Explicit token count for each subsequence
    std::vector<int32_t> past_lens;           // Explicit past length for each subsequence
    std::vector<int32_t> cache_intervals;     // Explicit cache interval for each subsequence
    int32_t qk_heads;                         // Number of query/key heads
    int32_t v_heads;                          // Number of value heads
    int32_t head_size;                        // Per-head hidden size (K and V dims in this test)
    ov::element::Type precision;              // Data precision for query/key/value/state tensors
};

struct paged_gated_delta_net_gpu_test : public ::testing::TestWithParam<paged_gated_delta_net_test_params> {
    tests::random_generator rg;

    template <typename T>
    static void normalize_and_scale(const T* src, size_t n, float scale, std::vector<float>& dst) {
        dst.resize(n);
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float v = static_cast<float>(src[i]);
            dst[i] = v;
            sum += v * v;
        }
        float inv = 1.0f / std::sqrt(sum + 1e-6f);
        for (size_t i = 0; i < n; i++) {
            dst[i] *= inv * scale;
        }
    }

    template <typename T>
    static void run_reference(const std::vector<T>& query,
                              const std::vector<T>& key,
                              const std::vector<T>& value,
                              const std::vector<T>& gate,
                              const std::vector<T>& beta,
                              std::vector<T>& recurrent_state_table,
                              const std::vector<int32_t>& subsequence_begins,
                              const std::vector<int32_t>& block_indices,
                              const std::vector<int32_t>& block_indices_begins,
                              const std::vector<int32_t>& past_lens,
                              const std::vector<int32_t>& cache_interval,
                              int32_t qk_heads,
                              int32_t v_heads,
                              int32_t qk_head_size,
                              int32_t v_head_size,
                              std::vector<T>& output) {
        const int32_t tokens = static_cast<int32_t>(query.size()) / (qk_heads * qk_head_size);

        OPENVINO_ASSERT(static_cast<int32_t>(key.size()) == tokens * qk_heads * qk_head_size, "Key tensor size does not match inferred qk head size");
        OPENVINO_ASSERT(static_cast<int32_t>(value.size()) == tokens * v_heads * v_head_size, "Value tensor size does not match provided v head size");
        OPENVINO_ASSERT(static_cast<int32_t>(gate.size()) == tokens * v_heads, "Gate tensor size does not match inferred token/head dimensions");
        OPENVINO_ASSERT(static_cast<int32_t>(beta.size()) == tokens * v_heads, "Beta tensor size does not match inferred token/head dimensions");

        output.resize(static_cast<size_t>(tokens) * v_heads * v_head_size);

        const auto state_off = [=](int32_t block, int32_t h, int32_t k_idx, int32_t v_idx) {
            return ((block * v_heads + h) * v_head_size + v_idx) * qk_head_size + k_idx;
        };

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(qk_head_size));
        const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
        const int32_t group_size = v_heads / qk_heads;

        OPENVINO_ASSERT(static_cast<int32_t>(recurrent_state_table.size()) % (v_heads * qk_head_size * v_head_size) == 0,
                        "State table size is inconsistent with inferred qk/v head sizes");

        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t token_begin = subsequence_begins[seq];
            const int32_t token_end = subsequence_begins[seq + 1];
            const int32_t block_begin = block_indices_begins[seq];
            const int32_t block_end = block_indices_begins[seq + 1];
            const int32_t seq_blocks = std::max(block_end - block_begin, 0);
            const int32_t past_len = past_lens[seq];
            const int32_t interval = cache_interval[seq];
            const int32_t prev_nums = (interval > 0) ? (past_len % interval) : 0;

            for (int32_t h = 0; h < v_heads; h++) {
                const int32_t hk = h / group_size;
                std::vector<float> state(static_cast<size_t>(qk_head_size) * v_head_size, 0.0f);

                const int32_t block_id = block_indices[block_begin];

                for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                    for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                        state[k_idx * v_head_size + v_idx] = static_cast<float>(recurrent_state_table[state_off(block_id, h, k_idx, v_idx)]);
                    }
                }

                for (int32_t token = token_begin; token < token_end; token++) {
                    const auto q_ptr = query.data() + (token * qk_heads + hk) * qk_head_size;
                    const auto k_ptr = key.data() + (token * qk_heads + hk) * qk_head_size;

                    std::vector<float> q_norm;
                    std::vector<float> k_norm;
                    normalize_and_scale(q_ptr, qk_head_size, attn_scale, q_norm);
                    normalize_and_scale(k_ptr, qk_head_size, 1.0f, k_norm);

                    const float b_g = std::exp(static_cast<float>(gate[token * v_heads + h]));
                    const float b_beta = static_cast<float>(beta[token * v_heads + h]);

                    for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                        const float b_v = static_cast<float>(value[(token * v_heads + h) * v_head_size + v_idx]);

                        float h_k = 0.0f;
                        for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                            auto& s = state[k_idx * v_head_size + v_idx];
                            s *= b_g;
                            h_k += s * k_norm[k_idx];
                        }

                        const float update = (b_v - h_k) * b_beta;
                        float out_v = 0.0f;
                        for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                            auto& s = state[k_idx * v_head_size + v_idx];
                            s += k_norm[k_idx] * update;
                            out_v += s * q_norm[k_idx];
                        }

                        output[(token * v_heads + h) * v_head_size + v_idx] = static_cast<T>(out_v);
                    }

                    const int32_t processed_tokens = (token - token_begin) + 1;
                    const int32_t cached_tokens = prev_nums + processed_tokens;
                    const bool reached_interval_boundary = (interval > 0) && ((cached_tokens % interval) == 0);
                    const bool reached_sequence_end = (token == token_end - 1);
                    if (reached_interval_boundary || reached_sequence_end) {
                        const int32_t slot = interval > 0 ? (1 + (cached_tokens - 1) / interval) : 1;
                        if (slot < seq_blocks) {
                            const int32_t block_id = block_indices[block_begin + slot];
                            for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                                for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                                    recurrent_state_table[state_off(block_id, h, k_idx, v_idx)] = static_cast<T>(state[k_idx * v_head_size + v_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void execute_t(const paged_gated_delta_net_test_params& p) {
        auto& engine = get_test_engine();

        const auto& subseq_tokens = p.subsequence_tokens;
        const auto& param_past_lens = p.past_lens;
        const auto& param_cache_intervals = p.cache_intervals;
        const int32_t num_sequences = static_cast<int32_t>(subseq_tokens.size());
        const int32_t tokens = static_cast<int32_t>(std::accumulate(subseq_tokens.begin(), subseq_tokens.end(), size_t{0}));
        const int32_t qk_heads = p.qk_heads;
        const int32_t v_heads = p.v_heads;
        const int32_t head_size = p.head_size;

        OPENVINO_ASSERT(num_sequences > 1, "Test should cover num_sequences > 1");
        OPENVINO_ASSERT(tokens >= num_sequences, "tokens should be >= num_sequences");
        OPENVINO_ASSERT(static_cast<int32_t>(param_past_lens.size()) == num_sequences, "past_lens size must match number of subsequences");
        OPENVINO_ASSERT(static_cast<int32_t>(param_cache_intervals.size()) == num_sequences, "cache_intervals size must match number of subsequences");

        std::vector<int32_t> subsequence_begins;
        std::vector<int32_t> past_lens;
        std::vector<int32_t> cache_interval;
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins;

        subsequence_begins.reserve(static_cast<size_t>(num_sequences + 1));
        block_indices_begins.reserve(static_cast<size_t>(num_sequences + 1));
        past_lens.reserve(static_cast<size_t>(num_sequences));
        cache_interval.reserve(static_cast<size_t>(num_sequences));

        subsequence_begins.push_back(0);
        block_indices_begins.push_back(0);

        int32_t acc_tokens = 0;
        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t seq_tokens = static_cast<int32_t>(subseq_tokens[seq]);
            OPENVINO_ASSERT(seq_tokens > 0, "Each subsequence must contain at least one token");
            acc_tokens += seq_tokens;
            subsequence_begins.push_back(acc_tokens);

            const int32_t seq_past_len = param_past_lens[seq];
            const int32_t seq_interval = param_cache_intervals[seq];
            OPENVINO_ASSERT(seq_past_len >= 0, "past_len must be non-negative");
            OPENVINO_ASSERT(seq_interval >= 0, "cache_interval must be >= 0");
            past_lens.push_back(seq_past_len);
            cache_interval.push_back(seq_interval);

            const int32_t required_slots = [&]() {
                if (seq_interval == 0) {
                    return 2;
                }

                const int32_t prev_nums = seq_past_len % seq_interval;
                const int32_t write_blocks = (prev_nums + seq_tokens + seq_interval - 1) / seq_interval;
                return 1 + write_blocks;
            }();
            for (int32_t i = 0; i < required_slots; i++) {
                block_indices.push_back(total_blocks + i);
            }
            total_blocks += required_slots;
            block_indices_begins.push_back(total_blocks);
        }

        layout q_layout({tokens, qk_heads, head_size}, p.precision, format::bfyx);
        layout k_layout({tokens, qk_heads, head_size}, p.precision, format::bfyx);
        layout v_layout({tokens, v_heads, head_size}, p.precision, format::bfyx);
        layout state_layout({static_cast<int32_t>(block_indices.size()), v_heads, head_size, head_size}, p.precision, format::bfyx);
        layout gate_layout({tokens, v_heads}, p.precision, format::bfyx);
        layout beta_layout({tokens, v_heads}, p.precision, format::bfyx);
        layout seq_begins_layout({static_cast<int32_t>(subsequence_begins.size())}, data_types::i32, format::bfyx);
        layout block_indices_layout({static_cast<int32_t>(block_indices.size())}, data_types::i32, format::bfyx);
        layout block_indices_begins_layout({static_cast<int32_t>(block_indices_begins.size())}, data_types::i32, format::bfyx);
        layout past_lens_layout({static_cast<int32_t>(past_lens.size())}, data_types::i32, format::bfyx);
        layout cache_interval_layout({static_cast<int32_t>(cache_interval.size())}, data_types::i32, format::bfyx);

        auto q_mem = engine.allocate_memory(q_layout);
        auto k_mem = engine.allocate_memory(k_layout);
        auto v_mem = engine.allocate_memory(v_layout);
        auto state_mem = engine.allocate_memory(state_layout);
        auto gate_mem = engine.allocate_memory(gate_layout);
        auto beta_mem = engine.allocate_memory(beta_layout);
        auto seq_begins_mem = engine.allocate_memory(seq_begins_layout);
        auto block_indices_mem = engine.allocate_memory(block_indices_layout);
        auto block_indices_begins_mem = engine.allocate_memory(block_indices_begins_layout);
        auto past_lens_mem = engine.allocate_memory(past_lens_layout);
        auto cache_interval_mem = engine.allocate_memory(cache_interval_layout);

        auto query = rg.generate_random_1d<T>(q_mem->count(), -1, 1, 127);
        auto key = rg.generate_random_1d<T>(k_mem->count(), -1, 1, 127);
        auto value = rg.generate_random_1d<T>(v_mem->count(), -1, 1, 127);
        auto gate = rg.generate_random_1d<T>(gate_mem->count(), -1, 1, 127);
        auto beta = rg.generate_random_1d<T>(beta_mem->count(), 0, 1, 127);
        auto state = rg.generate_random_1d<T>(state_mem->count(), -1, 1, 127);

        set_values(q_mem, query);
        set_values(k_mem, key);
        set_values(v_mem, value);
        set_values(gate_mem, gate);
        set_values(beta_mem, beta);
        set_values(state_mem, state);
        set_values(seq_begins_mem, subsequence_begins);
        set_values(block_indices_mem, block_indices);
        set_values(block_indices_begins_mem, block_indices_begins);
        set_values(past_lens_mem, past_lens);
        set_values(cache_interval_mem, cache_interval);

        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("state", state_layout));
        topo.add(input_layout("g", gate_layout));
        topo.add(input_layout("beta", beta_layout));
        topo.add(input_layout("subsequence_begins", seq_begins_layout));
        topo.add(input_layout("block_indices", block_indices_layout));
        topo.add(input_layout("block_indices_begins", block_indices_begins_layout));
        topo.add(input_layout("past_lens", past_lens_layout));
        topo.add(input_layout("cache_interval", cache_interval_layout));
        topo.add(paged_gated_delta_net("paged_gdn",
                                       {input_info("q"),
                                        input_info("k"),
                                        input_info("v"),
                                        input_info("state"),
                                        input_info("g"),
                                        input_info("beta"),
                                        input_info("subsequence_begins"),
                                        input_info("block_indices"),
                                        input_info("block_indices_begins"),
                                        input_info("past_lens"),
                                        input_info("cache_interval")}));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

        network->set_input_data("q", q_mem);
        network->set_input_data("k", k_mem);
        network->set_input_data("v", v_mem);
        network->set_input_data("state", state_mem);
        network->set_input_data("g", gate_mem);
        network->set_input_data("beta", beta_mem);
        network->set_input_data("subsequence_begins", seq_begins_mem);
        network->set_input_data("block_indices", block_indices_mem);
        network->set_input_data("block_indices_begins", block_indices_begins_mem);
        network->set_input_data("past_lens", past_lens_mem);
        network->set_input_data("cache_interval", cache_interval_mem);

        auto outputs = network->execute();
        auto out_mem = outputs.at("paged_gdn").get_memory();

        std::vector<T> ref_output;
        auto ref_state = state;
        run_reference(query,
                      key,
                      value,
                      gate,
                      beta,
                      ref_state,
                      subsequence_begins,
                      block_indices,
                      block_indices_begins,
                      past_lens,
                      cache_interval,
                      qk_heads,
                      v_heads,
                      head_size,
                      head_size,
                      ref_output);

        const float tol = std::is_same<T, ov::float16>::value ? 4e-2f : 1e-3f;

        {
            cldnn::mem_lock<T, mem_lock_type::read> out_lock(out_mem, get_test_stream());
            ASSERT_EQ(out_mem->count(), ref_output.size());
            for (size_t i = 0; i < ref_output.size(); i++) {
                ASSERT_NEAR(static_cast<float>(out_lock[i]), static_cast<float>(ref_output[i]), tol) << " at output index=" << i;
            }
        }

        {
            cldnn::mem_lock<T, mem_lock_type::read> state_lock(state_mem, get_test_stream());
            ASSERT_EQ(state_mem->count(), ref_state.size());
            for (size_t i = 0; i < ref_state.size(); i++) {
                ASSERT_NEAR(static_cast<float>(state_lock[i]), static_cast<float>(ref_state[i]), tol) << " at state index=" << i;
            }
        }
    }

    void execute(const paged_gated_delta_net_test_params& p) {
        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p);
            return;
        }

        if (p.precision == ov::element::f32) {
            execute_t<float>(p);
            return;
        }

        FAIL() << "Unsupported precision for paged_gated_delta_net test";
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<paged_gated_delta_net_test_params>& info) {
        std::string subseq_tokens_str;
        for (size_t i = 0; i < info.param.subsequence_tokens.size(); i++) {
            if (i > 0)
                subseq_tokens_str += "-";
            subseq_tokens_str += std::to_string(info.param.subsequence_tokens[i]);
        }

        std::string past_lens_str;
        for (size_t i = 0; i < info.param.past_lens.size(); i++) {
            if (i > 0)
                past_lens_str += "-";
            past_lens_str += std::to_string(info.param.past_lens[i]);
        }

        std::string cache_intervals_str;
        for (size_t i = 0; i < info.param.cache_intervals.size(); i++) {
            if (i > 0)
                cache_intervals_str += "-";
            cache_intervals_str += std::to_string(info.param.cache_intervals[i]);
        }

        std::string result = "paged_gated_delta_net_gpu_test_" + info.param.precision.to_string() + "_" + subseq_tokens_str + "_" + past_lens_str + "_" +
                             cache_intervals_str + "_" + std::to_string(info.param.qk_heads) + "_" + std::to_string(info.param.v_heads) + "_" +
                             std::to_string(info.param.head_size);
        return result;
    }
};

TEST_P(paged_gated_delta_net_gpu_test, basic) {
    execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_gated_delta_net_gpu_test,
                         paged_gated_delta_net_gpu_test,
                         // Parameter order:
                         // {subsequence_tokens, past_lens, cache_intervals, qk_heads, v_heads, head_size, precision}
                         ::testing::Values(
                             // f16: decode stage (past_len > 0), head_size 16/64/128, different sequence counts.
                             paged_gated_delta_net_test_params{{3, 3}, {2, 3}, {2, 3}, 2, 2, 16, ov::element::f16},
                             paged_gated_delta_net_test_params{{2, 3, 2}, {1, 2, 4}, {2, 3, 2}, 2, 2, 64, ov::element::f16},
                             paged_gated_delta_net_test_params{{1, 2, 3, 2}, {1, 3, 2, 4}, {2, 2, 3, 4}, 2, 2, 128, ov::element::f16},
                             // f16: prefill stage (all past_len = 0), head_size 16/64/128, different sequence counts.
                             paged_gated_delta_net_test_params{{3, 2}, {0, 0}, {2, 2}, 2, 2, 16, ov::element::f16},
                             paged_gated_delta_net_test_params{{2, 2, 2}, {0, 0, 0}, {2, 2, 2}, 2, 2, 64, ov::element::f16},
                             paged_gated_delta_net_test_params{{1, 2, 2, 3}, {0, 0, 0, 0}, {2, 2, 3, 2}, 2, 2, 128, ov::element::f16},
                             // f16: mixed stage (some past_len = 0, some past_len > 0), head_size 16/64/128.
                             paged_gated_delta_net_test_params{{4, 2}, {0, 3}, {2, 2}, 2, 2, 16, ov::element::f16},
                             paged_gated_delta_net_test_params{{1, 4, 2}, {0, 2, 0}, {2, 3, 2}, 2, 2, 64, ov::element::f16},
                             paged_gated_delta_net_test_params{{2, 1, 3, 2}, {0, 4, 0, 1}, {2, 2, 3, 2}, 2, 2, 128, ov::element::f16},
                             // f16: blocking stress (cache_interval=16), seq lengths occupy 1/2/3/4 blocks.
                             // fully occupied past blocks
                             paged_gated_delta_net_test_params{{16, 32, 48, 64}, {16, 32, 48, 64}, {16, 16, 16, 16}, 2, 2, 64, ov::element::f16},
                             // partially occupied past blocks
                             paged_gated_delta_net_test_params{{16, 32, 48, 64}, {1, 17, 33, 49}, {16, 16, 16, 16}, 2, 2, 64, ov::element::f16},

                             // f32: decode stage (past_len > 0), head_size 16/64/128, different sequence counts.
                             paged_gated_delta_net_test_params{{3, 3}, {2, 3}, {2, 3}, 2, 2, 16, ov::element::f32},
                             paged_gated_delta_net_test_params{{2, 3, 2}, {1, 2, 4}, {2, 3, 2}, 2, 2, 64, ov::element::f32},
                             paged_gated_delta_net_test_params{{1, 2, 3, 2}, {1, 3, 2, 4}, {2, 2, 3, 4}, 2, 2, 128, ov::element::f32},
                             // f32: prefill stage (all past_len = 0), head_size 16/64/128, different sequence counts.
                             paged_gated_delta_net_test_params{{3, 2}, {0, 0}, {2, 2}, 2, 2, 16, ov::element::f32},
                             paged_gated_delta_net_test_params{{2, 2, 2}, {0, 0, 0}, {2, 2, 2}, 2, 2, 64, ov::element::f32},
                             paged_gated_delta_net_test_params{{1, 2, 2, 3}, {0, 0, 0, 0}, {2, 2, 3, 2}, 2, 2, 128, ov::element::f32},
                             // f32: mixed stage (some past_len = 0, some past_len > 0), head_size 16/64/128.
                             paged_gated_delta_net_test_params{{4, 2}, {0, 3}, {2, 2}, 2, 2, 16, ov::element::f32},
                             paged_gated_delta_net_test_params{{1, 4, 2}, {0, 2, 0}, {2, 3, 2}, 2, 2, 64, ov::element::f32},
                             paged_gated_delta_net_test_params{{2, 1, 3, 2}, {0, 4, 0, 1}, {2, 2, 3, 2}, 2, 2, 128, ov::element::f32},
                             // f32: blocking stress (cache_interval=16), seq lengths occupy 1/2/3/4 blocks.
                             // fully occupied past blocks
                             paged_gated_delta_net_test_params{{16, 32, 48, 64}, {16, 32, 48, 64}, {16, 16, 16, 16}, 2, 2, 64, ov::element::f32},
                             // partially occupied past blocks
                             paged_gated_delta_net_test_params{{16, 32, 48, 64}, {1, 17, 33, 49}, {16, 16, 16, 16}, 2, 2, 64, ov::element::f32}),
                         paged_gated_delta_net_gpu_test::PrintToStringParamName);

}  // namespace
