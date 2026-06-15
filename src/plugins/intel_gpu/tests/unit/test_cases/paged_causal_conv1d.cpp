// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_causal_conv1d.hpp>
#include <vector>

#include "paged_causal_conv1d_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct paged_causal_conv1d_test_params {
    int32_t tokens;
    int32_t num_sequences;
    int32_t hidden_size;
    int32_t kernel_size;
    ov::element::Type precision;
    bool with_bias;
};

struct paged_causal_conv1d_gpu_test : public ::testing::TestWithParam<paged_causal_conv1d_test_params> {
    tests::random_generator rg;

    struct paging_desc {
        std::vector<int32_t> subsequence_begins;
        std::vector<int32_t> block_indices;
        std::vector<int32_t> block_indices_begins;
        std::vector<int32_t> past_lens;
        std::vector<int32_t> cache_interval;
        int32_t num_blocks = 0;
    };

    template <typename T>
    static T to_data_type(float v) {
        if constexpr (std::is_same<T, ov::float16>::value) {
            return ov::float16(v);
        }
        return static_cast<T>(v);
    }

    template <typename T>
    void load_input(cldnn::memory::ptr mem, const std::vector<T>& input_data) {
        set_values(mem, input_data);
    }

    static paging_desc make_paging_desc(int32_t tokens, int32_t num_sequences) {
        OPENVINO_ASSERT(num_sequences > 0, "num_sequences must be positive");
        OPENVINO_ASSERT(tokens >= num_sequences, "tokens must be >= num_sequences");

        paging_desc d;
        d.subsequence_begins.reserve(static_cast<size_t>(num_sequences + 1));
        d.block_indices_begins.reserve(static_cast<size_t>(num_sequences + 1));
        d.past_lens.reserve(static_cast<size_t>(num_sequences));
        d.cache_interval.reserve(static_cast<size_t>(num_sequences));

        d.subsequence_begins.push_back(0);
        d.block_indices_begins.push_back(0);

        int32_t consumed_tokens = 0;
        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t rem_tokens = tokens - consumed_tokens;
            const int32_t rem_seq = num_sequences - seq;
            const int32_t seq_tokens = rem_tokens / rem_seq;

            consumed_tokens += seq_tokens;
            d.subsequence_begins.push_back(consumed_tokens);

            const int32_t seq_past_len = 1 + (seq % 3);
            const int32_t seq_interval = 2 + (seq % 2);
            d.past_lens.push_back(seq_past_len);
            d.cache_interval.push_back(seq_interval);

            const int32_t prev_nums = seq_past_len % seq_interval;
            const int32_t total_cached = prev_nums + seq_tokens;
            const int32_t max_slot = 1 + (total_cached - 1) / seq_interval;
            const int32_t required_slots = 1 + std::max<int32_t>(1, max_slot);
            for (int32_t i = 0; i < required_slots; i++) {
                d.block_indices.push_back(total_blocks + i);
            }
            total_blocks += required_slots;
            d.block_indices_begins.push_back(total_blocks);
        }

        d.num_blocks = total_blocks;
        return d;
    }

    template <typename T>
    static void run_reference(const std::vector<T>& input_embeds,
                              const std::vector<T>& conv_weight,
                              const std::vector<T>& conv_bias,
                              std::vector<T>& conv_state_table,
                              const std::vector<int32_t>& subsequence_begins,
                              const std::vector<int32_t>& block_indices,
                              const std::vector<int32_t>& block_indices_begins,
                              const std::vector<int32_t>& past_lens,
                              const std::vector<int32_t>& cache_interval,
                              int32_t hidden_size,
                              int32_t kernel_size,
                              std::vector<T>& output_embeds) {
        const int32_t token_count = static_cast<int32_t>(input_embeds.size()) / hidden_size;
        output_embeds.resize(static_cast<size_t>(token_count) * hidden_size);

        auto state_off = [hidden_size, kernel_size](int32_t block, int32_t h, int32_t k) {
            return (block * hidden_size + h) * kernel_size + k;
        };

        for (int32_t seq = 0; seq < static_cast<int32_t>(subsequence_begins.size()) - 1; seq++) {
            const int32_t token_begin = subsequence_begins[seq];
            const int32_t token_end = subsequence_begins[seq + 1];
            const int32_t blk_begin = block_indices_begins[seq];
            const int32_t blk_end = block_indices_begins[seq + 1];

            if (token_begin < 0 || token_end < token_begin || blk_end <= blk_begin) {
                continue;
            }

            const int32_t block_span = blk_end - blk_begin;
            if (block_span <= 1) {
                continue;
            }

            const int32_t seq_interval = cache_interval[seq];
            const int32_t prev_nums = (seq_interval > 0) ? (past_lens[seq] % seq_interval) : 0;
            const int32_t seq_tokens = token_end - token_begin;
            const int32_t read_physical_block = block_indices[blk_begin];

            for (int32_t h = 0; h < hidden_size; h++) {
                std::vector<float> state(static_cast<size_t>(kernel_size), 0.0f);
                for (int32_t k = 0; k < kernel_size; k++) {
                    state[k] = static_cast<float>(conv_state_table[state_off(read_physical_block, h, k)]);
                }

                const float bias_val = conv_bias.empty() ? 0.0f : static_cast<float>(conv_bias[h]);

                for (int32_t t = 0; t < seq_tokens; t++) {
                    for (int32_t k = 0; k + 1 < kernel_size; k++) {
                        state[k] = state[k + 1];
                    }

                    const int32_t token_idx = token_begin + t;
                    state[kernel_size - 1] = static_cast<float>(input_embeds[token_idx * hidden_size + h]);

                    float sum = bias_val;
                    const int32_t w_base = h * kernel_size;
                    for (int32_t k = 0; k < kernel_size; k++) {
                        sum = std::fma(state[k], static_cast<float>(conv_weight[w_base + k]), sum);
                    }

                    output_embeds[token_idx * hidden_size + h] = to_data_type<T>(sum);

                    const int32_t cached_tokens = prev_nums + (t + 1);
                    const bool interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
                    const bool is_last_token = (t == seq_tokens - 1);
                    if (interval_hit || is_last_token) {
                        const int32_t slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                        if (slot >= 1 && slot < block_span) {
                            const int32_t physical_block = block_indices[blk_begin + slot];
                            for (int32_t k = 0; k < kernel_size; k++) {
                                conv_state_table[state_off(physical_block, h, k)] = to_data_type<T>(state[k]);
                            }
                        }
                    }
                }
            }
        }
    }

    topology create_topology(layout input_data_layout,
                             layout state_data_layout,
                             layout weight_data_layout,
                             layout bias_data_layout,
                             layout subseq_data_layout,
                             layout block_idx_data_layout,
                             layout block_idx_begins_data_layout,
                             layout past_lens_data_layout,
                             layout cache_interval_data_layout,
                             data_types output_dt) {
        topology topo;

        topo.add(input_layout("input_embeds", input_data_layout));
        topo.add(input_layout("conv_state_table", state_data_layout));
        topo.add(input_layout("conv_weight", weight_data_layout));
        topo.add(input_layout("conv_bias", bias_data_layout));
        topo.add(input_layout("subsequence_begins", subseq_data_layout));
        topo.add(input_layout("block_indices", block_idx_data_layout));
        topo.add(input_layout("block_indices_begins", block_idx_begins_data_layout));
        topo.add(input_layout("past_lens", past_lens_data_layout));
        topo.add(input_layout("cache_interval", cache_interval_data_layout));

        topo.add(paged_causal_conv1d("paged_causal_conv1d",
                                     {input_info("input_embeds"),
                                      input_info("conv_state_table"),
                                      input_info("conv_weight"),
                                      input_info("conv_bias"),
                                      input_info("subsequence_begins"),
                                      input_info("block_indices"),
                                      input_info("block_indices_begins"),
                                      input_info("past_lens"),
                                      input_info("cache_interval")}));

        topo.add(reorder("output", input_info("paged_causal_conv1d"), format::bfyx, output_dt));
        return topo;
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(topology& topo,
                                                                    cldnn::memory::ptr input_mem,
                                                                    cldnn::memory::ptr state_mem,
                                                                    cldnn::memory::ptr weight_mem,
                                                                    cldnn::memory::ptr bias_mem,
                                                                    cldnn::memory::ptr subseq_mem,
                                                                    cldnn::memory::ptr block_idx_mem,
                                                                    cldnn::memory::ptr block_idx_begins_mem,
                                                                    cldnn::memory::ptr past_lens_mem,
                                                                    cldnn::memory::ptr cache_interval_mem) {
        auto& engine = get_test_engine();

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), false);

        net->set_input_data("input_embeds", input_mem);
        net->set_input_data("conv_state_table", state_mem);
        net->set_input_data("conv_weight", weight_mem);
        net->set_input_data("conv_bias", bias_mem);
        net->set_input_data("subsequence_begins", subseq_mem);
        net->set_input_data("block_indices", block_idx_mem);
        net->set_input_data("block_indices_begins", block_idx_begins_mem);
        net->set_input_data("past_lens", past_lens_mem);
        net->set_input_data("cache_interval", cache_interval_mem);

        auto outputs = net->execute();
        auto output = outputs.at("output").get_memory();

        return std::make_tuple(output, net);
    }

    template <typename T>
    void execute_t(const paged_causal_conv1d_test_params& p, data_types data_type, float output_tolerance, float state_tolerance) {
        auto& engine = get_test_engine();

        const auto page = make_paging_desc(p.tokens, p.num_sequences);

        const layout input_layout({p.tokens, p.hidden_size}, data_type, format::bfyx);
        const layout state_layout({page.num_blocks, p.hidden_size, p.kernel_size}, data_type, format::bfyx);
        const layout weight_layout({p.hidden_size, 1, p.kernel_size}, data_type, format::bfyx);
        const layout bias_layout({p.with_bias ? p.hidden_size : 0}, data_type, format::bfyx);

        const layout subseq_layout({static_cast<int32_t>(page.subsequence_begins.size())}, data_types::i32, format::bfyx);
        const layout block_idx_layout({static_cast<int32_t>(page.block_indices.size())}, data_types::i32, format::bfyx);
        const layout block_idx_begins_layout({static_cast<int32_t>(page.block_indices_begins.size())}, data_types::i32, format::bfyx);
        const layout past_lens_layout({static_cast<int32_t>(page.past_lens.size())}, data_types::i32, format::bfyx);
        const layout cache_interval_layout({static_cast<int32_t>(page.cache_interval.size())}, data_types::i32, format::bfyx);

        auto input_mem = engine.allocate_memory(input_layout);
        auto state_mem = engine.allocate_memory(state_layout);
        auto weight_mem = engine.allocate_memory(weight_layout);
        auto bias_mem = engine.allocate_memory(bias_layout);
        auto subseq_mem = engine.allocate_memory(subseq_layout);
        auto block_idx_mem = engine.allocate_memory(block_idx_layout);
        auto block_idx_begins_mem = engine.allocate_memory(block_idx_begins_layout);
        auto past_lens_mem = engine.allocate_memory(past_lens_layout);
        auto cache_interval_mem = engine.allocate_memory(cache_interval_layout);

        auto input_embeds = rg.generate_random_1d<T>(ov::shape_size(input_layout.get_shape()), -1.0f, 1.0f, 256);
        auto conv_state_table = rg.generate_random_1d<T>(ov::shape_size(state_layout.get_shape()), -1.0f, 1.0f, 256);
        auto conv_weight = rg.generate_random_1d<T>(ov::shape_size(weight_layout.get_shape()), -1.0f, 1.0f, 256);
        std::vector<T> conv_bias;
        if (p.with_bias) {
            conv_bias = rg.generate_random_1d<T>(ov::shape_size(bias_layout.get_shape()), -0.5f, 0.5f, 256);
        }

        auto ref_state = conv_state_table;
        std::vector<T> ref_output;
        run_reference(input_embeds,
                      conv_weight,
                      conv_bias,
                      ref_state,
                      page.subsequence_begins,
                      page.block_indices,
                      page.block_indices_begins,
                      page.past_lens,
                      page.cache_interval,
                      p.hidden_size,
                      p.kernel_size,
                      ref_output);

        load_input(input_mem, input_embeds);
        load_input(state_mem, conv_state_table);
        load_input(weight_mem, conv_weight);
        if (!conv_bias.empty()) {
            load_input(bias_mem, conv_bias);
        }
        load_input(subseq_mem, page.subsequence_begins);
        load_input(block_idx_mem, page.block_indices);
        load_input(block_idx_begins_mem, page.block_indices_begins);
        load_input(past_lens_mem, page.past_lens);
        load_input(cache_interval_mem, page.cache_interval);

        auto topo = create_topology(input_layout,
                                    state_layout,
                                    weight_layout,
                                    bias_layout,
                                    subseq_layout,
                                    block_idx_layout,
                                    block_idx_begins_layout,
                                    past_lens_layout,
                                    cache_interval_layout,
                                    data_type);

        auto [out_mem, net] =
            run_network(topo, input_mem, state_mem, weight_mem, bias_mem, subseq_mem, block_idx_mem, block_idx_begins_mem, past_lens_mem, cache_interval_mem);

        ASSERT_TRUE(out_mem != nullptr);
        ASSERT_EQ(out_mem->count(), ref_output.size());

        {
            cldnn::mem_lock<T, mem_lock_type::read> out_data(out_mem, get_test_stream());
            for (size_t i = 0; i < ref_output.size(); i++) {
                ASSERT_NEAR(static_cast<float>(out_data[i]), static_cast<float>(ref_output[i]), output_tolerance) << " at index=" << i;
            }
        }

        ASSERT_EQ(state_mem->count(), ref_state.size());
        {
            cldnn::mem_lock<T, mem_lock_type::read> state_data(state_mem, get_test_stream());
            for (size_t i = 0; i < ref_state.size(); i++) {
                ASSERT_NEAR(static_cast<float>(state_data[i]), static_cast<float>(ref_state[i]), state_tolerance) << " at index=" << i;
            }
        }
    }

    void execute(const paged_causal_conv1d_test_params& p) {
        const auto cldnn_precision = cldnn::element_type_to_data_type(p.precision);

        if (p.precision == ov::element::f16) {
            execute_t<ov::float16>(p, cldnn_precision, 0.03f, 0.03f);
            return;
        }

        if (p.precision == ov::element::f32) {
            execute_t<float>(p, cldnn_precision, 1e-4f, 1e-4f);
            return;
        }

        FAIL() << "Unsupported precision for paged_causal_conv1d test";
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<paged_causal_conv1d_test_params>& info) {
        const auto& p = info.param;
        return "paged_causal_conv1d_gpu_test_" + p.precision.to_string() + "_tokens_" + std::to_string(p.tokens) + "_seq_" + std::to_string(p.num_sequences) +
               "_hidden_" + std::to_string(p.hidden_size) + "_kernel_" + std::to_string(p.kernel_size) + (p.with_bias ? "_bias" : "_no_bias");
    }
};

TEST_P(paged_causal_conv1d_gpu_test, basic) {
    const auto p = GetParam();
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(smoke_paged_causal_conv1d_gpu_test,
                         paged_causal_conv1d_gpu_test,
                         ::testing::Values(paged_causal_conv1d_test_params{8, 2, 16, 4, ov::element::f16, true},
                                           paged_causal_conv1d_test_params{12, 3, 32, 5, ov::element::f16, true},
                                           paged_causal_conv1d_test_params{8, 2, 16, 4, ov::element::f32, true},
                                           paged_causal_conv1d_test_params{12, 3, 32, 5, ov::element::f32, true},
                                           paged_causal_conv1d_test_params{8, 2, 16, 4, ov::element::f16, false},
                                           paged_causal_conv1d_test_params{8, 2, 16, 4, ov::element::f32, false}),
                         paged_causal_conv1d_gpu_test::PrintToStringParamName);

}  // namespace
