// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "cpu_parallel.hpp"
#include "gtest/gtest.h"
#include "nodes/kernels/linear_attn/recurrent_linear_attn.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace {

void normalize_and_scale(const float* src, size_t n, float scale, std::vector<float>& dst) {
    dst.resize(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
        sum += src[i] * src[i];
    }
    const float inv = 1.0f / std::sqrt(sum + 1e-6f);
    for (size_t i = 0; i < n; i++) {
        dst[i] *= inv * scale;
    }
}

void run_reference(const std::vector<float>& query,
                   const std::vector<float>& key,
                   const std::vector<float>& value,
                   const std::vector<float>& gate,
                   const std::vector<float>& beta,
                   std::vector<float>& recurrent_state_table,
                   const std::vector<int32_t>& subsequence_begins,
                   const std::vector<int32_t>& block_indices,
                   const std::vector<int32_t>& block_indices_begins,
                   [[maybe_unused]] const std::vector<int32_t>& past_lens,
                   const std::vector<int32_t>& cache_interval,
                   int32_t qk_heads,
                   int32_t v_heads,
                   int32_t qk_head_size,
                   int32_t v_head_size,
                   std::vector<float>& output) {
    const int32_t tokens = static_cast<int32_t>(query.size()) / (qk_heads * qk_head_size);
    const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
    const int32_t group_size = v_heads / qk_heads;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(qk_head_size));

    output.resize(static_cast<size_t>(tokens) * v_heads * v_head_size);

    const auto state_off = [=](int32_t block, int32_t h, int32_t k_idx, int32_t v_idx) {
        return ((block * v_heads + h) * qk_head_size + k_idx) * v_head_size + v_idx;
    };

    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t token_begin = subsequence_begins[seq];
        const int32_t token_end = subsequence_begins[seq + 1];
        const int32_t block_begin = block_indices_begins[seq];
        const int32_t block_end = block_indices_begins[seq + 1];
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        const int32_t seq_interval = cache_interval[seq];

        for (int32_t h = 0; h < v_heads; h++) {
            const int32_t hk = h / group_size;
            std::vector<float> state(static_cast<size_t>(qk_head_size) * v_head_size, 0.0f);

            if (seq_blocks > 0) {
                // Block 0 is the read source state for each subsequence.
                const int32_t block_id = block_indices[block_begin];
                for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                    for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                        state[k_idx * v_head_size + v_idx] =
                            recurrent_state_table[state_off(block_id, h, k_idx, v_idx)];
                    }
                }
            }

            for (int32_t token = token_begin; token < token_end; token++) {
                const auto q_ptr = query.data() + (token * qk_heads + hk) * qk_head_size;
                const auto k_ptr = key.data() + (token * qk_heads + hk) * qk_head_size;

                std::vector<float> q_norm;
                std::vector<float> k_norm;
                normalize_and_scale(q_ptr, qk_head_size, q_scale, q_norm);
                normalize_and_scale(k_ptr, qk_head_size, 1.0f, k_norm);

                const float b_g = std::exp(gate[token * v_heads + h]);
                const float b_beta = beta[token * v_heads + h];

                for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                    const float b_v = value[(token * v_heads + h) * v_head_size + v_idx];

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

                    output[(token * v_heads + h) * v_head_size + v_idx] = out_v;
                }

                if (seq_interval > 0 && seq_blocks > 0) {
                    // Blocking policy: block 0 is read-only seed state. We checkpoint newly
                    // generated state by local progress in this subsequence, so writes go to
                    // block 1,2,... at every interval boundary (and force one final flush).
                    const int32_t local_token_idx = token - token_begin;
                    const int32_t processed_tokens = local_token_idx + 1;
                    const bool should_store = ((processed_tokens % seq_interval) == 0) || (token == token_end - 1);
                    if (should_store) {
                        // Write newly generated cache to blocks 1..N.
                        const int32_t slot = (processed_tokens + seq_interval - 1) / seq_interval;
                        if (slot < seq_blocks) {
                            const int32_t block_id = block_indices[block_begin + slot];
                            for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                                for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                                    recurrent_state_table[state_off(block_id, h, k_idx, v_idx)] =
                                        state[k_idx * v_head_size + v_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

using RecurrentLinearAttnPagedParams =
    std::tuple<int32_t,
               int32_t,
               int32_t,
               int32_t,
               std::vector<int32_t>,
               std::vector<int32_t>>;  // qk_heads, v_heads, qk_head_size, v_head_size, seq_lengths, cache_intervals

class PagedGatedDeltaNetKernelTest : public testing::TestWithParam<RecurrentLinearAttnPagedParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RecurrentLinearAttnPagedParams>& obj) {
        const auto& [qk_heads, v_heads, qk_head_size, v_head_size, seq_lengths, cache_intervals] = obj.param;
        std::ostringstream result;
        result << "QKHeads=" << qk_heads;
        result << "_VHeads=" << v_heads;
        result << "_QKHeadSize=" << qk_head_size;
        result << "_VHeadSize=" << v_head_size;
        result << "_SeqLens=";
        for (size_t i = 0; i < seq_lengths.size(); i++) {
            if (i > 0)
                result << "x";
            result << seq_lengths[i];
        }
        result << "_Intervals=";
        for (size_t i = 0; i < cache_intervals.size(); i++) {
            if (i > 0)
                result << "x";
            result << cache_intervals[i];
        }
        return result.str();
    }
};

TEST_P(PagedGatedDeltaNetKernelTest, MatchesReferenceAndUpdatesState) {
    const auto& [qk_heads, v_heads, qk_head_size, v_head_size, seq_lengths, cache_intervals_input] = GetParam();

    ASSERT_GT(qk_heads, 0);
    ASSERT_GT(v_heads, 0);
    ASSERT_EQ(v_heads % qk_heads, 0);
    ASSERT_GT(qk_head_size, 0);
    ASSERT_GT(v_head_size, 0);
    ASSERT_FALSE(seq_lengths.empty());
    ASSERT_EQ(seq_lengths.size(), cache_intervals_input.size());
    for (const auto len : seq_lengths) {
        ASSERT_GT(len, 0);
    }
    for (const auto interval : cache_intervals_input) {
        ASSERT_GT(interval, 0);
    }

    const int32_t num_sequences = static_cast<int32_t>(seq_lengths.size());
    const int32_t tokens = std::accumulate(seq_lengths.begin(), seq_lengths.end(), 0);
    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<int32_t> past_lens;
    std::vector<int32_t> cache_interval;

    subsequence_begins.reserve(static_cast<size_t>(num_sequences + 1));
    block_indices_begins.reserve(static_cast<size_t>(num_sequences + 1));
    past_lens.reserve(static_cast<size_t>(num_sequences));
    cache_interval.reserve(static_cast<size_t>(num_sequences));

    subsequence_begins.push_back(0);
    block_indices_begins.push_back(0);

    int32_t total_blocks = 0;
    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t seq_len = seq_lengths[seq];
        subsequence_begins.push_back(subsequence_begins.back() + seq_len);

        const int32_t seq_past_len = 1 + (seq % 3);
        const int32_t seq_interval = cache_intervals_input[seq];
        past_lens.push_back(seq_past_len);
        cache_interval.push_back(seq_interval);

        // One extra slot is required for block 0 (read-only init state), then we need
        // ceil(seq_len / seq_interval) writable slots for checkpoints into blocks 1..N.
        const int32_t required_slots = 1 + (seq_len + seq_interval - 1) / seq_interval;
        for (int32_t i = 0; i < required_slots; i++) {
            block_indices.push_back(total_blocks + i);
        }
        total_blocks += required_slots;
        block_indices_begins.push_back(total_blocks);
    }

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto make_vec = [&](size_t n) {
        std::vector<float> v(n);
        std::generate(v.begin(), v.end(), [&]() {
            return dist(gen);
        });
        return v;
    };

    auto query = make_vec(static_cast<size_t>(tokens) * qk_heads * qk_head_size);
    auto key = make_vec(static_cast<size_t>(tokens) * qk_heads * qk_head_size);
    auto value = make_vec(static_cast<size_t>(tokens) * v_heads * v_head_size);
    auto gate = make_vec(static_cast<size_t>(tokens) * v_heads);
    auto beta = make_vec(static_cast<size_t>(tokens) * v_heads);

    auto recurrent_state_table =
        make_vec(static_cast<size_t>(block_indices.size()) * v_heads * qk_head_size * v_head_size);

    std::vector<float> output(static_cast<size_t>(tokens) * v_heads * v_head_size, 0.0f);

    PlainTensor query_t;
    query_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(qk_heads), static_cast<size_t>(qk_head_size)},
                   query.data());
    PlainTensor key_t;
    key_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(qk_heads), static_cast<size_t>(qk_head_size)},
                 key.data());
    PlainTensor value_t;
    value_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(v_heads), static_cast<size_t>(v_head_size)},
                   value.data());
    PlainTensor state_t;
    state_t.resize({block_indices.size(),
                    static_cast<size_t>(v_heads),
                    static_cast<size_t>(qk_head_size),
                    static_cast<size_t>(v_head_size)},
                   recurrent_state_table.data());
    PlainTensor gate_t;
    gate_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(v_heads)}, gate.data());
    PlainTensor beta_t;
    beta_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(v_heads)}, beta.data());

    PlainTensor subsequence_begins_t;
    subsequence_begins_t.resize({subsequence_begins.size()}, const_cast<int32_t*>(subsequence_begins.data()));
    PlainTensor block_indices_t;
    block_indices_t.resize({block_indices.size()}, const_cast<int32_t*>(block_indices.data()));
    PlainTensor block_indices_begins_t;
    block_indices_begins_t.resize({block_indices_begins.size()}, const_cast<int32_t*>(block_indices_begins.data()));
    PlainTensor past_lens_t;
    past_lens_t.resize({past_lens.size()}, const_cast<int32_t*>(past_lens.data()));
    PlainTensor cache_interval_t;
    cache_interval_t.resize({cache_interval.size()}, const_cast<int32_t*>(cache_interval.data()));

    PlainTensor output_t;
    output_t.resize({static_cast<size_t>(tokens), static_cast<size_t>(v_heads), static_cast<size_t>(v_head_size)},
                    output.data());

    std::shared_ptr<CpuParallel> cpu_parallel(
        new CpuParallel(ov::intel_cpu::TbbPartitioner::STATIC, static_cast<size_t>(CpuParallel::default_multiplier)));
    std::vector<float> temp_buffer(static_cast<size_t>(cpu_parallel->get_num_worker_threads()) * 3 * qk_head_size,
                                   0.0f);

    std::vector<float> ref_output;
    auto ref_state = recurrent_state_table;
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
                  qk_head_size,
                  v_head_size,
                  ref_output);

    recurrent_linear_attn_paged(query_t,
                                key_t,
                                value_t,
                                state_t,
                                gate_t,
                                beta_t,
                                subsequence_begins_t,
                                block_indices_t,
                                block_indices_begins_t,
                                past_lens_t,
                                cache_interval_t,
                                output_t,
                                temp_buffer.data(),
                                cpu_parallel);

    constexpr float tol = 1e-4f;
    for (size_t i = 0; i < output.size(); i++) {
        ASSERT_NEAR(output[i], ref_output[i], tol) << "Output mismatch at index " << i;
    }
    for (size_t i = 0; i < recurrent_state_table.size(); i++) {
        ASSERT_NEAR(recurrent_state_table[i], ref_state[i], tol) << "State mismatch at index " << i;
    }
}

const std::vector<RecurrentLinearAttnPagedParams> params = {
    {2, 4, 8, 8, {3, 3}, {2, 3}},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}},
};

INSTANTIATE_TEST_SUITE_P(RecurrentLinearAttnPagedKernelUnitTest,
                         PagedGatedDeltaNetKernelTest,
                         ::testing::ValuesIn(params),
                         PagedGatedDeltaNetKernelTest::getTestCaseName);

}  // namespace