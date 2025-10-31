// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <numeric>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/moe_scatter_reduction.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename T, typename TN>
std::vector<T> flatten (const std::vector<std::vector<TN>>& vec2d) {
    return std::accumulate(vec2d.begin(), vec2d.end(), std::vector<T>{},
        [](auto& a, auto& b) { a.insert(a.end(), b.begin(), b.end()); return a; });
}

template <typename T, typename ShapeType>
auto create_layout(ShapeType shape) {
  if constexpr (std::is_same_v<T, float>)
      return layout{shape, data_types::f32, format::bfyx};
  else
      return layout{shape, data_types::f16, format::bfyx};
}

template <typename T>
void test_moe_scatter_reduction(bool is_caching_test, size_t k) {
    auto& engine = get_test_engine();
    // num_tokens 30
    // hidden_size 64
    // num total experts 32
    // num_active_experts_per_token 2
    // num_actual_used_experts 7
    // input0 activation [30, 64]
    // input1 experts_info_offset [7]
    // input2 tokens_per_expert [30*2*64]
    size_t num_tokens = 30;
    size_t num_total_experts = 32;
    size_t hidden_size = k;
    size_t num_active_experts_per_token = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = create_layout<T>(input_activation_shape);

    auto experts_per_token_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_per_token_layout = layout{experts_per_token_shape, data_types::i32, format::bfyx};

    auto expert_weights_per_token_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto expert_weights_per_token_layout = create_layout<T>(expert_weights_per_token_shape);

    auto experts_info_offsets_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_info_offsets_layout = layout{experts_info_offsets_shape, data_types::i32, format::bfyx};

    auto tokens_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto tokens_per_expert_layout = layout{tokens_per_expert_shape, data_types::i32, format::bfyx};

    auto tokens_len_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto tokens_len_per_expert_layout = layout{tokens_len_per_expert_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        input_layout("experts_per_token", experts_per_token_layout),
        input_layout("expert_weights_per_token", expert_weights_per_token_layout),
        input_layout("tokens_per_expert", tokens_per_expert_layout),
        input_layout("experts_info_offsets", experts_info_offsets_layout),
        input_layout("tokens_len_per_expert", tokens_len_per_expert_layout),
        moe_scatter_reduction("moe_scatter_reduction", input_info("input"), input_info("experts_per_token"),
            input_info("expert_weights_per_token"), input_info("tokens_per_expert"), input_info("experts_info_offsets"),
            input_info("tokens_len_per_expert"), num_active_experts_per_token)
    );
    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
    auto input_data_layout = create_layout<T>(input_data_shape);

    std::vector<T> input_data;
    for (size_t i = 0; i < num_tokens * num_active_experts_per_token; ++i) {
        for (size_t h = 0; h < hidden_size; ++h)
            input_data.push_back(static_cast<T>(i));
    }
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    // topk result
    std::vector<std::vector<size_t>> experts_per_token = {{0, 5},  {5, 7},   {0, 10}, {11, 20}, {7, 10},  {0, 7},   {20, 31}, {11, 31}, {11, 20}, {7, 10},
                                          {0, 5},  {11, 31}, {0, 7},  {0, 20},  {10, 31}, {10, 20}, {7, 31},  {0, 31},  {5, 31},  {7, 31},
                                          {7, 20}, {0, 10},  {0, 5},  {5, 11},  {7, 11},  {5, 31},  {7, 31},  {0, 31},  {0, 10},  {11, 20}};

    std::vector<std::vector<T>> expert_weights_per_token = {{1.0, .9}, {.8, .7}, {.6, .5}, {.4, .3}, {.2, .1}, {.9, .8}, {.7, .6}, {.5, .4}, {.3, .2}, {.1, .0},
                                          {1.0, .9}, {.8, .7}, {.6, .5}, {.4, .3}, {.2, .1}, {.9, .8}, {.7, .6}, {.5, .4}, {.3, .2}, {.1, .0},
                                          {1.0, .9}, {.8, .7}, {.6, .5}, {.4, .3}, {.2, .1}, {.9, .8}, {.7, .6}, {.5, .4}, {.3, .2}, {.1, .0}};

    std::vector<std::vector<size_t>> tokens_per_expert_tmp(num_total_experts, std::vector<size_t>{});

    for (size_t i = 0; i < experts_per_token.size(); ++i) {
        for (size_t j = 0; j < experts_per_token[i].size(); ++j)
            tokens_per_expert_tmp[experts_per_token[i][j]].push_back(i);
    }

    std::vector<int32_t> tokens_per_expert_data;
    std::vector<int32_t> tokens_len_per_expert_data;

    for (size_t i = 0; i < tokens_per_expert_tmp.size(); ++i) {
        tokens_len_per_expert_data.push_back(tokens_per_expert_tmp[i].size());
        if (tokens_per_expert_tmp[i].empty())
            continue;
        for (size_t j = 0; j < tokens_per_expert_tmp[i].size(); ++j) {
            tokens_per_expert_data.push_back(tokens_per_expert_tmp[i][j]);
        }
    }

    std::vector<int32_t> expert_info_start_idx(tokens_len_per_expert_data.size());
    std::exclusive_scan(tokens_len_per_expert_data.begin(), tokens_len_per_expert_data.end(),
        expert_info_start_idx.begin(), 0);

    // tokens per expert
    // experts 0, 5, 7, 10, 11, 20, 31 are used
    // experts[0] offset  : 0  {0, 2, 5, 10, 12, 13, 17, 21, 22, 27, 28}
    // experts[5] offset  : 11 {0, 1, 10, 18, 22, 23, 25}
    // experts[7] offset  : 18 {1, 4, 5, 9, 12, 16, 19, 20, 24, 26}
    // experts[10] offset : 28 {2, 4, 9, 14, 15, 21, 28}
    // experts[11] offset : 35 {3, 7, 8, 11, 23, 24, 29}
    // experts[20] offset : 42 {3, 6, 8, 13, 15, 20, 29}
    // experts[31] offset : 49 {6, 7, 11, 14, 16, 17, 18, 19, 25, 26, 27}

    auto experts_per_token_data_shape    = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(num_active_experts_per_token)};
    auto experts_per_token_data_layout   = layout{experts_per_token_data_shape, data_types::i32, format::bfyx};
    auto experts_per_token_data_mem      = engine.allocate_memory(experts_per_token_data_layout);
    set_values(experts_per_token_data_mem,  flatten<int32_t>(experts_per_token));

    auto expert_weights_per_token_data_shape    = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(num_active_experts_per_token)};
    auto expert_weights_per_token_data_layout   = create_layout<T>(expert_weights_per_token_data_shape);
    auto expert_weights_per_token_data_mem      = engine.allocate_memory(expert_weights_per_token_data_layout);
    set_values(expert_weights_per_token_data_mem, flatten<T>(expert_weights_per_token));

    auto tokens_per_expert_data_shape    = ov::PartialShape{ov::Dimension(tokens_per_expert_data.size())};
    auto tokens_per_expert_data_layout   = layout{tokens_per_expert_data_shape, data_types::i32, format::bfyx};
    auto tokens_per_expert_data_mem      = engine.allocate_memory(tokens_per_expert_data_layout);
    set_values(tokens_per_expert_data_mem, tokens_per_expert_data);

    auto expert_info_offsets_data_shape    = ov::PartialShape{ov::Dimension(expert_info_start_idx.size())};
    auto expert_info_offsets_data_layout   = layout{expert_info_offsets_data_shape, data_types::i32, format::bfyx};
    auto expert_info_offsets_data_mem      = engine.allocate_memory(expert_info_offsets_data_layout);
    set_values(expert_info_offsets_data_mem, expert_info_start_idx);

    auto tokens_len_per_expert_data_shape    = ov::PartialShape{ov::Dimension(tokens_len_per_expert_data.size())};
    auto tokens_len_per_expert_data_layout   = layout{expert_info_offsets_data_shape, data_types::i32, format::bfyx};
    auto tokens_len_per_expert_data_mem      = engine.allocate_memory(tokens_len_per_expert_data_layout);
    set_values(tokens_len_per_expert_data_mem, tokens_len_per_expert_data);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config, is_caching_test);
    network.set_input_data("input", input_mem);
    network.set_input_data("experts_per_token", experts_per_token_data_mem);
    network.set_input_data("expert_weights_per_token", expert_weights_per_token_data_mem);
    network.set_input_data("tokens_per_expert", tokens_per_expert_data_mem);
    network.set_input_data("experts_info_offsets", expert_info_offsets_data_mem);
    network.set_input_data("tokens_len_per_expert", tokens_len_per_expert_data_mem);
    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<T> ref_output;
    for (size_t i = 0; i < num_tokens; ++i) {
        std::vector<T> token_output(hidden_size, 0.0f);
        for (size_t j = 0; j < num_active_experts_per_token; j++) {
            size_t expert_idx = experts_per_token[i][j];
            float expert_weight = expert_weights_per_token[i][j];
            for (size_t k = 0; k < tokens_per_expert_tmp[expert_idx].size(); k++) {
                if (i == tokens_per_expert_tmp[expert_idx][k]) {
                    size_t input_idx = expert_info_start_idx[expert_idx] + k;
                    // copy out the data and multiply the weight then add to token output
                    std::vector<T> token_data(hidden_size);
                    std::copy(input_data.begin() + input_idx * hidden_size, input_data.begin() + (input_idx + 1)*hidden_size, token_data.begin());
                    std::transform(token_data.begin(), token_data.end(), token_data.begin(), [&expert_weight](auto& c){return c*expert_weight;});
                    std::transform(token_data.begin(), token_data.end(), token_output.begin(), token_output.begin(), std::plus<T>());
                    break;
                }
            }
        }
        ref_output.insert(ref_output.end(), token_output.begin(), token_output.end());
    }

    for (size_t i = 0; i < num_tokens * hidden_size; i++) {
       EXPECT_NEAR(ref_output[i], output_ptr[i], 1e-1);
    }
}

TEST(moe_unit, moe_scatter_reduction_test_one_batch_aligned_f32) {
    test_moe_scatter_reduction<float>(false, 64);
}

TEST(moe_unit, moe_scatter_reduction_test_one_batch_unaligned_f32) {
    test_moe_scatter_reduction<float>(false, 66);
}

TEST(moe_unit, moe_scatter_reduction_test_multi_batch_aligned_f32) {
    test_moe_scatter_reduction<float>(false, 2880);
}

TEST(moe_unit, moe_scatter_reduction_test_multi_batch_unaligned_f32) {
    test_moe_scatter_reduction<float>(false, 2882);
}

TEST(moe_unit, moe_scatter_reduction_test_one_batch_aligned_f16) {
    test_moe_scatter_reduction<ov::float16>(false, 64);
}
TEST(moe_unit, moe_scatter_reduction_test_one_batch_unaligned_f16) {
    test_moe_scatter_reduction<ov::float16>(false, 66);
}

TEST(moe_unit, moe_scatter_reduction_test_multi_batch_aligned_f16) {
    test_moe_scatter_reduction<ov::float16>(false, 2880);
}

TEST(moe_unit, moe_scatter_reduction_test_multi_batch_unaligned_f16) {
    test_moe_scatter_reduction<ov::float16>(false, 2882);
}

TEST(moe_unit, moe_scatter_reduction_test_cached) {
    test_moe_scatter_reduction<float>(true, 64);
}
