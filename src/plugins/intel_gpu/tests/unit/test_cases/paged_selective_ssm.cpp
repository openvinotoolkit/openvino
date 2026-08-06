// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_selective_ssm.hpp>
#include <numeric>
#include <vector>

#include "paged_selective_ssm_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct paged_selective_ssm_test_params {
    std::vector<int32_t> seq_tokens;
    std::vector<int32_t> processed_tokens;
    std::vector<int32_t> cache_intervals;
    int32_t num_heads;
    int32_t num_groups;
    int32_t head_dim;
    int32_t state_size;
    ov::element::Type precision;
};

struct paged_selective_ssm_gpu_test : public ::testing::TestWithParam<paged_selective_ssm_test_params> {
    tests::random_generator rg;

    template <typename T>
    static void run_reference(const std::vector<T>& A,
                              const std::vector<T>& dt,
                              const std::vector<T>& B,
                              const std::vector<T>& x,
                              const std::vector<T>& C,
                              std::vector<T>& state,
                              const std::vector<int32_t>& subsequence_begins,
                              const std::vector<int32_t>& block_indices,
                              const std::vector<int32_t>& block_indices_begins,
                              const std::vector<int32_t>& num_processed_tokens,
                              const std::vector<int32_t>& cache_interval,
                              int32_t num_heads,
                              int32_t num_groups,
                              int32_t head_dim,
                              int32_t state_size,
                              std::vector<T>& output) {
        const int32_t heads_per_group = num_heads / num_groups;
        const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
        const int32_t tokens = static_cast<int32_t>(x.size()) / (num_heads * head_dim);
        output.resize(static_cast<size_t>(tokens) * num_heads * head_dim);
        const auto state_off = [=](int32_t block, int32_t h, int32_t p, int32_t n) {
            return ((block * num_heads + h) * head_dim + p) * state_size + n;
        };

        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t token_begin = subsequence_begins[seq];
            const int32_t token_end = subsequence_begins[seq + 1];
            const int32_t block_begin = block_indices_begins[seq];
            const int32_t block_end = block_indices_begins[seq + 1];
            const int32_t seq_blocks = std::max(block_end - block_begin, 0);
            const int32_t processed = num_processed_tokens[seq];
            const int32_t interval = cache_interval[seq];
            const int32_t prev_nums = interval > 0 ? (processed % interval) : 0;
            const int32_t first_block = block_indices[block_begin];

            for (int32_t h = 0; h < num_heads; h++) {
                const int32_t g = h / heads_per_group;
                for (int32_t p = 0; p < head_dim; p++) {
                    std::vector<float> local_state(static_cast<size_t>(state_size), 0.f);
                    for (int32_t n = 0; n < state_size; n++) {
                        local_state[n] = static_cast<float>(state[state_off(first_block, h, p, n)]);
                    }

                    for (int32_t token = token_begin; token < token_end; token++) {
                        const float dt_val = static_cast<float>(dt[token * num_heads + h]);
                        const float dA = std::exp(static_cast<float>(A[h]) * dt_val);
                        const float x_val = static_cast<float>(x[(token * num_heads + h) * head_dim + p]);
                        float acc = 0.f;
                        for (int32_t n = 0; n < state_size; n++) {
                            local_state[n] = local_state[n] * dA + x_val * dt_val * static_cast<float>(B[(token * num_groups + g) * state_size + n]);
                            acc += local_state[n] * static_cast<float>(C[(token * num_groups + g) * state_size + n]);
                        }
                        output[(token * num_heads + h) * head_dim + p] = static_cast<T>(acc);

                        const int32_t processed_now = (token - token_begin) + 1;
                        const int32_t cached_tokens = prev_nums + processed_now;
                        const bool reached_interval_boundary = interval > 0 && ((cached_tokens % interval) == 0);
                        const bool reached_sequence_end = token == token_end - 1;
                        if (reached_interval_boundary || reached_sequence_end) {
                            const int32_t slot = interval > 0 ? (1 + (cached_tokens - 1) / interval) : 1;
                            if (slot < seq_blocks) {
                                const int32_t block_id = block_indices[block_begin + slot];
                                for (int32_t n = 0; n < state_size; n++) {
                                    state[state_off(block_id, h, p, n)] = static_cast<T>(local_state[n]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void execute_t(const paged_selective_ssm_test_params& p) {
        auto& engine = get_test_engine();
        const auto data_type = cldnn::element_type_to_data_type(p.precision);
        const int32_t num_sequences = static_cast<int32_t>(p.seq_tokens.size());
        const int32_t tokens = static_cast<int32_t>(std::accumulate(p.seq_tokens.begin(), p.seq_tokens.end(), 0));

        std::vector<int32_t> subsequence_begins{0};
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins{0};
        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            subsequence_begins.push_back(subsequence_begins.back() + p.seq_tokens[seq]);
            int32_t required_slots = 2;
            if (p.cache_intervals[seq] > 0) {
                const int32_t prev_nums = p.processed_tokens[seq] % p.cache_intervals[seq];
                const int32_t write_blocks = (prev_nums + p.seq_tokens[seq] + p.cache_intervals[seq] - 1) / p.cache_intervals[seq];
                required_slots = 1 + write_blocks;
            }
            for (int32_t i = 0; i < required_slots; i++)
                block_indices.push_back(total_blocks + i);
            total_blocks += required_slots;
            block_indices_begins.push_back(total_blocks);
        }

        layout A_layout({p.num_heads}, data_type, format::bfyx);
        layout dt_layout({tokens, p.num_heads}, data_type, format::bfyx);
        layout BC_layout({tokens, p.num_groups, p.state_size}, data_type, format::bfyx);
        layout x_layout({tokens, p.num_heads, p.head_dim}, data_type, format::bfyx);
        layout state_layout({total_blocks, p.num_heads, p.head_dim, p.state_size}, data_type, format::bfyx);
        layout i32_vec_layout({static_cast<int32_t>(subsequence_begins.size())}, data_types::i32, format::bfyx);
        layout block_indices_layout({static_cast<int32_t>(block_indices.size())}, data_types::i32, format::bfyx);
        layout processed_layout({num_sequences}, data_types::i32, format::bfyx);

        auto A_mem = engine.allocate_memory(A_layout);
        auto dt_mem = engine.allocate_memory(dt_layout);
        auto B_mem = engine.allocate_memory(BC_layout);
        auto x_mem = engine.allocate_memory(x_layout);
        auto C_mem = engine.allocate_memory(BC_layout);
        auto state_mem = engine.allocate_memory(state_layout);
        auto subseq_mem = engine.allocate_memory(i32_vec_layout);
        auto blocks_mem = engine.allocate_memory(block_indices_layout);
        auto block_begins_mem = engine.allocate_memory(i32_vec_layout);
        auto processed_mem = engine.allocate_memory(processed_layout);
        auto interval_mem = engine.allocate_memory(processed_layout);

        auto A_data = rg.generate_random_1d<T>(A_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.2f), 256);
        auto dt_data = rg.generate_random_1d<T>(dt_mem->count(), static_cast<T>(0.f), static_cast<T>(0.5f), 256);
        auto B_data = rg.generate_random_1d<T>(B_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto x_data = rg.generate_random_1d<T>(x_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto C_data = rg.generate_random_1d<T>(C_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);
        auto state_data = rg.generate_random_1d<T>(state_mem->count(), static_cast<T>(-0.5f), static_cast<T>(0.5f), 256);

        set_values(A_mem, A_data);
        set_values(dt_mem, dt_data);
        set_values(B_mem, B_data);
        set_values(x_mem, x_data);
        set_values(C_mem, C_data);
        set_values(state_mem, state_data);
        set_values(subseq_mem, subsequence_begins);
        set_values(blocks_mem, block_indices);
        set_values(block_begins_mem, block_indices_begins);
        set_values(processed_mem, p.processed_tokens);
        set_values(interval_mem, p.cache_intervals);

        topology topo;
        topo.add(input_layout("A", A_layout));
        topo.add(input_layout("dt", dt_layout));
        topo.add(input_layout("B", BC_layout));
        topo.add(input_layout("x", x_layout));
        topo.add(input_layout("C", BC_layout));
        topo.add(input_layout("state", state_layout));
        topo.add(input_layout("subsequence_begins", i32_vec_layout));
        topo.add(input_layout("block_indices", block_indices_layout));
        topo.add(input_layout("block_indices_begins", i32_vec_layout));
        topo.add(input_layout("num_processed_tokens", processed_layout));
        topo.add(input_layout("cache_interval", processed_layout));
        topo.add(paged_selective_ssm("paged_selective_ssm",
                                     {input_info("A"),
                                      input_info("dt"),
                                      input_info("B"),
                                      input_info("x"),
                                      input_info("C"),
                                      input_info("state"),
                                      input_info("subsequence_begins"),
                                      input_info("block_indices"),
                                      input_info("block_indices_begins"),
                                      input_info("num_processed_tokens"),
                                      input_info("cache_interval")}));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);
        network->set_input_data("A", A_mem);
        network->set_input_data("dt", dt_mem);
        network->set_input_data("B", B_mem);
        network->set_input_data("x", x_mem);
        network->set_input_data("C", C_mem);
        network->set_input_data("state", state_mem);
        network->set_input_data("subsequence_begins", subseq_mem);
        network->set_input_data("block_indices", blocks_mem);
        network->set_input_data("block_indices_begins", block_begins_mem);
        network->set_input_data("num_processed_tokens", processed_mem);
        network->set_input_data("cache_interval", interval_mem);
        auto outputs = network->execute();

        std::vector<T> ref_output;
        auto ref_state = state_data;
        run_reference(A_data,
                      dt_data,
                      B_data,
                      x_data,
                      C_data,
                      ref_state,
                      subsequence_begins,
                      block_indices,
                      block_indices_begins,
                      p.processed_tokens,
                      p.cache_intervals,
                      p.num_heads,
                      p.num_groups,
                      p.head_dim,
                      p.state_size,
                      ref_output);

        auto out_mem = outputs.at("paged_selective_ssm").get_memory();
        cldnn::mem_lock<T, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
        cldnn::mem_lock<T, mem_lock_type::read> state_ptr(state_mem, get_test_stream());
        const float tol = p.precision == ov::element::f32 ? 1e-4f : 0.03f;
        for (size_t i = 0; i < ref_output.size(); i++) {
            ASSERT_NEAR(static_cast<float>(out_ptr[i]), static_cast<float>(ref_output[i]), tol) << "output idx=" << i;
        }
        for (size_t i = 0; i < ref_state.size(); i++) {
            ASSERT_NEAR(static_cast<float>(state_ptr[i]), static_cast<float>(ref_state[i]), tol) << "state idx=" << i;
        }
    }

    void execute(const paged_selective_ssm_test_params& p) {
        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p);
        } else {
            execute_t<float>(p);
        }
    }
};

TEST_P(paged_selective_ssm_gpu_test, basic) {
    execute(GetParam());
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_selective_ssm_gpu_test,
                         paged_selective_ssm_gpu_test,
                         ::testing::Values(paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::f32},
                                           paged_selective_ssm_test_params{{2, 4, 1}, {1, 2, 3}, {1, 3, 2}, 4, 1, 8, 16, ov::element::f32},
                                           paged_selective_ssm_test_params{{3, 2}, {1, 2}, {2, 0}, 4, 2, 8, 8, ov::element::f16}));

}  // namespace
