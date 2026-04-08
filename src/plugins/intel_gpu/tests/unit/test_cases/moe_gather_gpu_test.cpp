// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <numeric>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include "intel_gpu/op/moe_compressed.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;


// moe_gather
template <typename T>
void test_moe_gather(bool is_caching_test, int k) {
    auto& engine = get_test_engine();
    // num_tokens 30
    // hidden_size 64
    // num total experts 32
    // experts_per_token 2
    // num_actual_used_experts 7
    // input0 activation [30, 64]
    // input1 experts_info_offset [7]
    // input2 tokens_per_expert [30*2*64]
    size_t num_tokens = 30;
    size_t num_total_experts = 32;
    size_t hidden_size = k;
    int32_t num_experts_per_token = 2;

    ov::intel_gpu::op::MOECompressed::Config moe_config;
    moe_config.top_k = 2;
    moe_config.hidden_size = k;
    moe_config.num_expert = num_total_experts;
    moe_config.has_batch_dim = false;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto tokens_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto tokens_per_expert_layout = layout{tokens_per_expert_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        input_layout("tokens_per_expert", tokens_per_expert_layout),
        moe_gather("moe_gather", input_info("input"), input_info("tokens_per_expert"), moe_config)
    );
    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};

    std::vector<ov::float16> input_data;
    for (size_t i = 0; i < num_tokens; ++i) {
        for (size_t h = 0; h < hidden_size; ++h)
            input_data.push_back(static_cast<ov::float16>(i));
    }
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    // topk result
    std::vector<std::vector<int32_t>> experts_per_token = {{0, 5},  {5, 7},   {0, 10}, {11, 20}, {7, 10},  {0, 7},   {20, 31}, {11, 31}, {11, 20}, {7, 10},
                                          {0, 5},  {11, 31}, {0, 7},  {0, 20},  {10, 31}, {10, 20}, {7, 31},  {0, 31},  {5, 31},  {7, 31},
                                          {7, 20}, {0, 10},  {0, 5},  {5, 11},  {7, 11},  {5, 31},  {7, 31},  {0, 31},  {0, 10},  {11, 20}};

    std::vector<std::vector<int32_t>> tokens_per_expert_tmp(num_total_experts, std::vector<int32_t>{});

    for (size_t i = 0; i < experts_per_token.size(); ++i) {
        for (size_t j = 0; j < experts_per_token[i].size(); ++j)
            tokens_per_expert_tmp[experts_per_token[i][j]].push_back(static_cast<int32_t>(i));
    }

    std::vector<int32_t> tokens_per_expert_data;

    for (size_t i = 0; i < tokens_per_expert_tmp.size(); ++i) {
        if (tokens_per_expert_tmp[i].empty())
            continue;
        for (size_t j = 0; j < tokens_per_expert_tmp[i].size(); ++j) {
            tokens_per_expert_data.push_back(tokens_per_expert_tmp[i][j]);
        }
    }
    // experts 0, 5, 7, 10, 11, 20, 31 are used
    // experts[0] offset  : 0  {0, 2, 5, 10, 12, 13, 17, 21, 22, 27, 28}
    // experts[5] offset  : 11 {0, 1, 10, 18, 22, 23, 25}
    // experts[7] offset  : 18 {1, 4, 5, 9, 12, 16, 19, 20, 24, 26}
    // experts[10] offset : 28 {2, 4, 9, 14, 15, 21, 28}
    // experts[11] offset : 35 {3, 7, 8, 11, 23, 24, 29}
    // experts[20] offset : 42 {3, 6, 8, 13, 15, 20, 29}
    // experts[31] offset : 49 {6, 7, 11, 14, 16, 17, 18, 19, 25, 26, 27}

    auto tokens_per_expert_data_shape    = ov::PartialShape{ov::Dimension(tokens_per_expert_data.size())};
    auto tokens_per_expert_data_layout   = layout{tokens_per_expert_data_shape, data_types::i32, format::bfyx};
    auto tokens_per_expert_data_mem      = engine.allocate_memory(tokens_per_expert_data_layout);
    set_values(tokens_per_expert_data_mem, tokens_per_expert_data);


    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
          {"moe_gather", {format::bfyx, "", impl_types::ocl}}}));

    network network(engine, topology, config, is_caching_test);
    network.set_input_data("input", input_mem);
    network.set_input_data("tokens_per_expert", tokens_per_expert_data_mem);
    auto outputs = network.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::vector<ov::float16> ref_output;
    for (size_t i = 0; i < tokens_per_expert_data.size(); ++i) {
        int32_t token_id = tokens_per_expert_data[i];
        for (size_t h = 0; h < hidden_size; ++h) {
            ref_output.push_back(input_data[token_id * hidden_size + h]);
        }
    }
    for (size_t i = 0; i < num_tokens * num_experts_per_token * hidden_size; i++) {
        ASSERT_EQ(ref_output[i], output_ptr[i]);
    }
}

TEST(moe_unit, moe_gather_test_single_batch_aligned) {
    test_moe_gather<float>(false, 64);
}

TEST(moe_unit, moe_gather_test_single_batch_noaligned) {
    test_moe_gather<float>(false, 66);
}

TEST(moe_unit, moe_gather_test_multi_batch_aligned) {
    test_moe_gather<float>(false, 5120);
}

TEST(moe_unit, moe_gather_test_multi_batch_unaligned) {
    test_moe_gather<float>(false, 5124);
}

TEST(moe_unit, moe_gather_test_cached) {
    test_moe_gather<float>(true, 64);
}