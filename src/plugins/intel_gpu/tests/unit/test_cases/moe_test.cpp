// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_mask_gen.hpp>
#include <intel_gpu/primitives/moe_gemm.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"

using namespace cldnn;
using namespace ::tests;


TEST(moe_unit, moe_mask_gen_test) {
    auto& engine = get_test_engine();

    // num total experts 32
    // num active experts 2
    // input activation [30, 64]
    // topk [30, 2]
    // output expert_info_offsets [32]
    // output tokens_indices_per_expert [30*2]

    std::vector<int32_t> topk_idx = {
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
        4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4 ,8,
    };

    int64_t num_tokens = 30;
    int64_t num_active_experts_per_token = 2;
    int32_t num_actually_used_experts = 2;
    // input shape
    auto topk_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension(num_active_experts_per_token)};
    auto topk_layout = layout{topk_shape, data_types::i32, format::bfyx};

    std::vector<int32_t> num_actually_used_experts_ref = {num_actually_used_experts};
    topology topology(
        input_layout("input_topk", topk_layout),
        moe_mask_gen("moe_mask_gen", input_info("input_topk"), 32, 2),
        moe_mask_gen_reshape("moe_mask_gen_reshape",
                             input_info("moe_mask_gen", 0),
                             input_info("moe_mask_gen", 1),
                             input_info("moe_mask_gen", 2),
                             input_info("moe_mask_gen", 3),
                             input_info("moe_mask_gen", 4)),
        reorder("num_actual_used_experts",
                input_info("moe_mask_gen", moe_mask_gen::MoEMaskGenOutputIdx::NUM_ACTUALLY_USED_EXPERTS),
                format::bfyx,
                data_types::f32),
        reorder("tokens_per_experts", input_info("moe_mask_gen_reshape", moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT), format::bfyx, data_types::f32),
        reorder("experts_info_start_idx", input_info("moe_mask_gen_reshape", moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX), format::bfyx, data_types::f32),
        reorder("experts_ids", input_info("moe_mask_gen_reshape", moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID), format::bfyx, data_types::f32),
        reorder("tokens_lens_per_expert",
                input_info("moe_mask_gen_reshape", moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
                format::bfyx,
                data_types::f32));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    auto topk_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(num_active_experts_per_token)};
    auto topk_data_layout = layout{topk_data_shape, data_types::i32, format::bfyx};
    auto topk_idx_mem = engine.allocate_memory(topk_data_layout);
    set_values(topk_idx_mem, topk_idx);
    network->set_input_data("input_topk", topk_idx_mem);

    auto outputs = network->execute();
    const auto& output_num_actual_experts = outputs.at("num_actual_used_experts").get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_num_actual_experts_ptr(output_num_actual_experts, get_test_stream());
    ASSERT_EQ(static_cast<int32_t>(output_num_actual_experts_ptr[0]), num_actually_used_experts);

    const auto& output_tokens_per_experts = outputs.at("tokens_per_experts").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_tokens_per_experts_ptr(output_tokens_per_experts, get_test_stream());
    std::vector<float> tokens_per_expert_ref = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    for (size_t i = 0; i < static_cast<size_t>(num_tokens * num_active_experts_per_token); i++) {
        ASSERT_EQ(output_tokens_per_experts_ptr[i], tokens_per_expert_ref[i]);
    }
    std::vector<float> expert_ids_ref = {4, 8};
    const auto& output_expert_ids = outputs.at("experts_ids").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_expert_ids_ptr(output_expert_ids, get_test_stream());
    for (size_t i = 0; i < static_cast<size_t>(num_actually_used_experts); i++) {
        ASSERT_EQ(static_cast<int32_t>(output_expert_ids_ptr[i]), expert_ids_ref[i]);
    }
    std::vector<float> experts_info_start_idx_ref = {0, 30};
    const auto& output_experts_info_start_idx = outputs.at("experts_info_start_idx").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_experts_info_start_idx_ptr(output_experts_info_start_idx, get_test_stream());
    for (size_t i = 0; i < static_cast<size_t>(num_actually_used_experts); i++) {
        ASSERT_EQ(static_cast<int32_t>(output_experts_info_start_idx_ptr[i]), experts_info_start_idx_ref[i]);
    }
    std::vector<float> tokens_lens_per_expert_ref = {30, 30};
    const auto& output_tokens_lens_per_expert = outputs.at("tokens_lens_per_expert").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_tokens_lens_per_expert_ptr(output_tokens_lens_per_expert, get_test_stream());
    for (size_t i = 0; i < static_cast<size_t>(num_actually_used_experts); i++) {
        ASSERT_EQ(static_cast<int32_t>(output_tokens_lens_per_expert_ptr[i]), tokens_lens_per_expert_ref[i]);
    }
}

static std::vector<ov::float16> get_input_data(size_t M, size_t K) {
    std::vector<ov::float16> input_data(M * K);
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            input_data[m * K + k] = static_cast<ov::float16>((k % 7 + 1.0f) / 10);
        }
    }
    return input_data;
};

static std::vector<ov::float16> get_ref_moe_gemm(std::vector<ov::float16>& input, std::vector<ov::float16>& experts,
                                    size_t M, size_t K, size_t N,
                                    std::vector<int32_t>& experts_ids, std::vector<int32_t>& input_offset_per_expert,
                                    std::vector<int32_t>& input_tokens_lens,
                                    size_t num_experts_per_token,
                                    bool is_prefill = true) {
    std::vector<ov::float16> output(M * num_experts_per_token * N, 0.0f);
    size_t input_stride = K;
    size_t expert_stride = K * N;
    size_t output_stride = N;
    for (size_t i = 0; i < input_offset_per_expert.size(); i++) {
        int32_t expert_id = experts_ids[i];
        int32_t input_offset = 0;
        if (is_prefill)
            input_offset = input_offset_per_expert[i] * input_stride;
        int32_t weight_offset = expert_id * expert_stride;
        int32_t output_offset = input_offset_per_expert[i] * output_stride;
        int32_t tokens_lens = input_tokens_lens[i];
        for (int32_t m = 0; m < tokens_lens; m++) {
            for (size_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += input[input_offset + m * input_stride + k] * experts[weight_offset + n * K + k];
                }
                output[output_offset + m * output_stride + n] = sum;
            }
        }
    }
    return output;
};

static std::vector<ov::float16> get_f16_weight(size_t num_total_experts, size_t expert_out_N, size_t hidden_size) {
    std::vector<ov::float16> experts_data_f16(num_total_experts * hidden_size * expert_out_N);
    // create and quantize data
    for (size_t e = 0; e < num_total_experts; ++e) {
        for (size_t n = 0; n < expert_out_N ; ++n) {
            for (size_t h = 0; h < hidden_size; ++h) {
                size_t idx = e * expert_out_N * hidden_size + n * hidden_size + h;
                experts_data_f16[idx] = static_cast<ov::float16>((e + (n % 3) + (h % 5) + 1) / 10.0f);
            }
        }
    }
    return experts_data_f16;
}
TEST(moe_unit, moe_gemm_test_small_f16) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    // num total experts 32
    // num active experts 2
    // input activation [30, 64]
    // mask_gather_info
    //    expert_info_offsets [32]
    //    tokens_indices_per_expert [30*2]
    size_t num_tokens = 10;
    size_t hidden_size = 16;
    size_t num_total_experts = 4;
    size_t experts_out_N = 16;
    int32_t num_experts_per_token = 2;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    // weight to fill with 1.0f for initial test
    auto experts_data_f16 = get_f16_weight(num_total_experts, experts_out_N, hidden_size);
    set_values(experts_mem, experts_data_f16);

    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    topology topology(
        input_layout("input", input_activation_layout),
        data("moe_experts", experts_mem),
        input_layout("experts_ids", experts_ids_layout),
        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
        input_layout("input_tokens_lens", input_tokens_lens_layout),
        moe_gemm("moe_gemm", input_info("input"),
                             input_info("moe_experts"),
                             input_info("experts_ids"),
                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
                             input_info("input_tokens_lens"),
                             num_experts_per_token
        )
    );

    std::vector<int32_t> input_tokens_lens  = {6, 12, 2};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_experts_per_token), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data = get_input_data(num_tokens * num_experts_per_token, hidden_size);
    set_values(input_mem, input_data);

    std::vector<int32_t> experts_ids_data = {0, 1, 3};
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 6, 12};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    network.set_input_data("experts_ids", experts_ids_mem);
    network.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network.set_input_data("input_tokens_lens", input_tokens_lens_mem);

    auto outputs = network.execute();
    auto output_ref = get_ref_moe_gemm(input_data, experts_data_f16,
                                    num_tokens, hidden_size, experts_out_N,
                                    experts_ids_data, input_offset_per_expert_data,
                                    input_tokens_lens, num_experts_per_token);

    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    for (size_t m = 0; m < num_tokens; m++) {
        for (size_t n = 0; n < experts_out_N; n++) {
            ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.1f);
        }
    }
}

TEST(moe_unit, moe_gemm_test_large) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        auto& engine = get_test_engine();
        size_t num_tokens = 100;
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = 16;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(num_active_experts_per_token * num_tokens * hidden_size);
        int cur_token_base = 0;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            for (int len = 0; len < input_tokens_lens[i]; ++len) {
                for (size_t h = 0; h < hidden_size; ++h) {
                    input_data[(cur_token_base + len) * hidden_size + h] = static_cast<ov::float16>((i + 1) / 10.0f);
                }
            }
            cur_token_base += input_tokens_lens[i];
        }

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                //            std::cout << "c[" << m << "][" << n << "]: " << (float)output_ptr[m * experts_out_N + n] << std::endl;
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 100;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
        //network.set_input_data("input", input_prim);
    }
}

TEST(moe_unit, moe_gemm_test_generate_up) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        auto& engine = get_test_engine();
        size_t num_tokens = 1;
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = num_active_experts_per_token;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
            std::cout << "input_tokens_lens[" << i << "] : " << input_tokens_lens[i] << std::endl;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(1 * hidden_size);

        for (size_t h = 0; h < hidden_size; ++h) {
            input_data[h] = static_cast<ov::float16>((num_tokens) / 10.0f);
        }
        std::cout << "input shape : " << input_data_shape.to_string() << std::endl;

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
            std::cout << "input_offset_per_data[" << i << "] : " << input_offset_per_expert_data[i] << std::endl;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        std::cout << "input_offset_per_expert_data_shape : " << input_offset_per_expert_data_shape.to_string() << std::endl;
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token, 
                                           false);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 1;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
    }
}

TEST(moe_unit, moe_gemm_test_generate_down) {
    tests::random_generator rg(GET_SUITE_NAME);
    {
        size_t num_tokens = 1;
        auto& engine = get_test_engine();
        size_t hidden_size = 512;
        size_t num_total_experts = 32;
        size_t experts_out_N = 1024;
        int32_t num_active_experts_per_token = 4;
        int32_t num_actual_used_experts = num_active_experts_per_token;

        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::f16, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        // weight to fill with 1.0f for initial test
        std::vector<ov::float16> experts_data(num_total_experts * hidden_size * experts_out_N, 1.0f);
        set_values(experts_mem, experts_data);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension(num_total_experts)};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        topology topology(input_layout("input", input_activation_layout),
                          data("moe_experts", experts_mem),
                          input_layout("experts_ids", experts_ids_layout),
                          input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                          input_layout("input_tokens_lens", input_tokens_lens_layout),
                          moe_gemm("moe_gemm",
                                   input_info("input"),
                                   input_info("moe_experts"),
                                   input_info("experts_ids"),
                                   input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                   input_info("input_tokens_lens"),
                                   num_active_experts_per_token));
        // 16 experts used
        // 25 tokens per expert
        int num_tokens_per_expert = (num_active_experts_per_token * num_tokens) / num_actual_used_experts;
        std::vector<int32_t> input_tokens_lens(num_total_experts, -1);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_tokens_lens[i] = num_tokens_per_expert;
            std::cout << "input_tokens_lens[" << i << "] : " << input_tokens_lens[i] << std::endl;
        }

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(num_active_experts_per_token * hidden_size);

        for (int e = 0; e < num_active_experts_per_token; ++e) {
            for (size_t h = 0; h < hidden_size; ++h) {
                input_data[e * hidden_size + h] = static_cast<ov::float16>((1 + num_tokens * e) / 10.0f);
            }
        }
        std::cout << "input shape : " << input_data_shape.to_string() << std::endl;

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(num_total_experts, -1);
        int exp_stride = num_total_experts / num_actual_used_experts;
        for (int i = 0; i < num_actual_used_experts; ++i) {
            experts_ids_data[i] = i * exp_stride;
            std::cout << "experts_ids_data [" << i << "] : " << experts_ids_data[i] << std::endl;
        }

        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(num_actual_used_experts, 0);
        for (int32_t i = 0; i < num_actual_used_experts; ++i) {
            input_offset_per_expert_data[i] = num_tokens_per_expert * i;
            std::cout << "input_offset_per_data[" << i << "] : " << input_offset_per_expert_data[i] << std::endl;
        }

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        std::cout << "input_offset_per_expert_data_shape : " << input_offset_per_expert_data_shape.to_string() << std::endl;
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);
        network->set_input_data("input", input_mem);
        network->set_input_data("experts_ids", experts_ids_mem);
        network->set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network->set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network->execute();
        auto output_ref = get_ref_moe_gemm(input_data,
                                           experts_data,
                                           num_tokens,
                                           hidden_size,
                                           experts_out_N,
                                           experts_ids_data,
                                           input_offset_per_expert_data,
                                           input_tokens_lens,
                                           num_active_experts_per_token, 
                                           true);

        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t m = 0; m < num_tokens * num_active_experts_per_token; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], output_ref[m * experts_out_N + n], 0.001f);
            }
        }
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 1;
        int32_t num_experts = 32;
        int32_t hidden_size = 512;
        int32_t N = 1024;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto weights_prim = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::f16, format::bfyx});
        auto input = input_layout("input", input_activation_layout);
        auto w_data = data("weights", weights_prim);
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", 3, 3);
        topology topology;
        topology.add(input);
        topology.add(w_data);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
    }
}

static void quantize_u4(std::vector<ov::float16>& weight_fp, std::vector<uint8_t>& weight_u4, int B, int N, int K, int group_size,
                 std::vector<ov::float16>& weight_scale, std::vector<ov::float16>& weight_zp) {
    const uint8_t u4_max = 15;
    const uint8_t u4_min = 0;
    const int K_u4 = K/2;
    const int num_elements_per_byte = 2;
    const int num_scale_groups = K / group_size;

    for (int b = 0; b < B; b++) {
        for (int m = 0; m < N; m++) {
            int group_iter = 0;
            while (group_iter * group_size < K) {
                ov::float16 amax = std::numeric_limits<ov::float16>::min();
                ov::float16 amin = std::numeric_limits<ov::float16>::max();
                for (int ki = 0; ki < group_size; ki++) {
                    ov::float16 v = weight_fp[b * N * K + m * K + group_iter * group_size + ki];
                    amax = std::max(amax, v);
                    amin = std::min(amin, v);
                }
                float range = (float)amax - (float)amin;
                if (range <= 1e-5f)
                    range = 1e-2f;
                float inv_scale = (u4_max - u4_min) / range;
                float zp_tmp = (float) (u4_min - amin * inv_scale);
                ov::float16 zp = zp_tmp;
                // quantize
                for (int ki = 0; ki < group_size / num_elements_per_byte; ki++)  {
                    ov::float16 v0 = weight_fp[(b * N * K) + (m * K) + (group_iter * group_size) + num_elements_per_byte * ki];
                    ov::float16 v1 = weight_fp[(b * N * K) + (m * K) + (group_iter * group_size) + num_elements_per_byte * ki + 1];
                    uint8_t q0 = std::min(std::max((uint8_t)(float(v0) * inv_scale + (float)zp), (uint8_t)0), u4_max); // u4
                    uint8_t q1 = std::min(std::max((uint8_t)(float(v1) * inv_scale + (float)zp), (uint8_t)0), u4_max); // u4
    
                    uint8_t q0q1 = (q1 << 4) | (q0 & 0x0F);
                    weight_u4[(b * N * K_u4) + (m * K_u4) + (group_iter * group_size / num_elements_per_byte) + ki] = uint8_t(q0q1);
                }
                ov::float16 scale = 1 / inv_scale;
                weight_scale[b * N + m * num_scale_groups + group_iter ] = scale;
                weight_zp[b * N + m * num_scale_groups + group_iter] = zp;
                // test quantized result
//                for (int ki = 0; ki < group_size / num_elements_per_byte; ki++)
//                {
//                    uint8_t q_v = weight_u4[(b * N * K_u4) + (m * K_u4) + (group_iter * group_size / num_elements_per_byte) + ki];
//                    uint8_t q0 = q_v & 0x0F;
//                    uint8_t q1 = (q_v >> 4) & 0x0F;
//                    float dq0 = (float(q0) - float(zp)) * float(scale);
//                    float dq1 = (float(q1) - float(zp)) * float(scale);
//                    auto orig_idx = (b * N * K) + (m * K) + group_iter * group_size + ki * num_elements_per_byte;
//                    std::cout << "A[" << b << "][" << m << "][" << group_iter * group_size + ki * num_elements_per_byte     << "] (" << orig_idx << ") scale : " << scale << " zp : " << zp << " fp : " << float(weight_fp[orig_idx]) << " q: " << int(q0) << " dq: " << dq0 << std::endl;
//                    std::cout << "A[" << b << "][" << m << "][" << group_iter * group_size + ki * num_elements_per_byte + 1 << "] (" << orig_idx + 1 << ") scale : " << scale << " zp : " << zp << " fp : " << float(weight_fp[orig_idx + 1]) << " q: " << int(q1) << " dq: " << dq1 << std::endl;
//                }
                group_iter++;
            }
        }
    }
}

static void quantize_i4_sym(std::vector<ov::float16>& weight_fp, std::vector<uint8_t>& weight_i4, int B, int N, int K, int group_size, std::vector<ov::float16>& weight_scale) {
    const int8_t i4_max = 7;
    const int8_t i4_min = -8;
    const int K_i4 = K/2;
    const int num_elements_per_byte = 2;
    const int num_scale_groups = K / group_size;

    for (int b = 0; b < B; b++) {
        for (int m = 0; m < N; m++) {
            int group_iter = 0;
            while (group_iter * group_size < K) {
                ov::float16 amax = std::numeric_limits<ov::float16>::lowest();
                ov::float16 amin = std::numeric_limits<ov::float16>::max();
                for (int ki = 0; ki < group_size; ki++) {
                    ov::float16 v = weight_fp[b * N * K + m * K + group_iter * group_size + ki];
                    amax = std::max(amax, v);
                    amin = std::min(amin, v);
                }
                float abs_max = std::max(std::abs(float(amax)), std::abs(float(amin)));
                float inv_scale = (float)i4_max / abs_max;
                // quantize
                for (int ki = 0; ki < group_size / num_elements_per_byte; ki++)  {
                    ov::float16 v0 = weight_fp[(b * N * K) + (m * K) + (group_iter * group_size) + num_elements_per_byte * ki];
                    ov::float16 v1 = weight_fp[(b * N * K) + (m * K) + (group_iter * group_size) + num_elements_per_byte * ki + 1];
                    int8_t q0 = std::min(std::max((int8_t)(std::round(float(v0) * inv_scale)), i4_min), i4_max); // u4
                    int8_t q1 = std::min(std::max((int8_t)(std::round(float(v1) * inv_scale)), i4_min), i4_max); // u4
    
//                    uint8_t q0q1 = (q1 << 4) | (q0 & 0x0F);
                    uint8_t q0q1 = ((uint8_t)q1 << 4) | ((uint8_t)q0 & 0x0F);
                    weight_i4[b * N * K_i4 + (m * K_i4) + (group_iter * group_size / num_elements_per_byte) + ki] = q0q1;
                }
                ov::float16 scale = 1 / inv_scale;
                weight_scale[b * N + m * num_scale_groups + group_iter ] = scale;
                // test quantized result
                #if 0
                for (int ki = 0; ki < group_size / num_elements_per_byte; ki++)
                {
                    uint8_t q_v = weight_i4[(b * N * K_i4) + (m * K_i4) + (group_iter * group_size / num_elements_per_byte) + ki];
                    std::cout << "original u8 q: " << std::hex << (uint32_t) q_v << std::dec << std::endl;
                    int8_t q0 = (int8_t)(q_v & 0x0F);
                    int8_t q1 = (int8_t)((q_v >> 4) & 0x0F);
                    if (q0 > 7)
                        q0 -= 16;
                    if (q1 > 7)
                        q1 -= 16;
                    float dq0 = float(q0) * float(scale);
                    float dq1 = float(q1) * float(scale);
                    auto orig_idx = (b * N * K) + (m * K) + group_iter * group_size + ki * num_elements_per_byte;
                    std::cout << "A[" << b << "][" << m << "][" << group_iter * group_size + ki * num_elements_per_byte << "] (" << orig_idx
                              << ") scale : " << scale << " fp : " << float(weight_fp[orig_idx]) << " q: " << int(q0) << "(" << std::hex << int(q0) << ")" << std::dec
                              << " dq: " << dq0 << std::endl;
                    std::cout << "A[" << b << "][" << m << "][" << group_iter * group_size + ki * num_elements_per_byte + 1 << "] (" << orig_idx + 1
                              << ") scale : " << scale << " fp : " << float(weight_fp[orig_idx + 1]) << " q: " << int(q1) << " dq: " << dq1 << std::endl;
                }
                #endif
                group_iter++;
            }
        }
    }
}
static void reference_u4(const std::vector<uint8_t> &W, const std::vector<ov::float16> &In, std::vector<ov::float16> &C,
               const std::vector<int32_t> &experts_ids, const std::vector<int32_t> &input_offset_per_expert,
               const std::vector<int32_t> &input_tokens_lens,
               int32_t N, int32_t K,
               const std::vector<ov::float16> &W_scale, const std::vector<ov::float16> &W_zp, int32_t W_group_size)
{
    auto ld_w = K/2, ld_in = K, ld_out = N;
    auto batch = input_offset_per_expert.size();

    auto expert_stride = ld_w * N;
    std::cout << "expert_stride : " << expert_stride << std::endl;
    for (size_t b = 0; b < batch; b++) {
        int32_t expert_id = experts_ids[b];
        std::cout << "expert_id : " << expert_id << std::endl;
        auto Wp = &W[expert_id * expert_stride];
        auto Inp = &In[input_offset_per_expert[b] * ld_in];
        auto Cp = &C[input_offset_per_expert[b] * ld_out];
        auto cur_m = input_tokens_lens[b];

        for (int j = 0; j < cur_m; j++) {
            for (int n = 0; n < N; n++) {
                auto W_r = Wp + n * ld_w;
                auto In_r = Inp + j * ld_in;
                float acc = 0.0f;
                for (int k = 0; k < ld_w; k++) {
                    // decompress
                    uint8_t q0 = ((uint8_t)W_r[k] & 0x0F);
                    uint8_t q1 = ((uint8_t)W_r[k] >> 4) & 0x0F;
                    float scale = float(W_scale[expert_id * N + n]);
                    float zp = float(W_zp[expert_id * N + n]);
                    float fa0 = (float(q0) - zp) * scale;
                    float fa1 = (float(q1) - zp) * scale;
                    acc += fa0 * In_r[2 * k];
                    acc += fa1 * In_r[2 * k + 1];
                }
                Cp[j * ld_out + n] = static_cast<ov::float16>(acc);
            }
        }
    }
}
#if 1
static void reference_i4(const std::vector<uint8_t> &W, const std::vector<ov::float16> &In, std::vector<float> &C,
               const std::vector<int32_t> &experts_ids, const std::vector<int32_t> &input_offset_per_expert,
               const std::vector<int32_t> &input_tokens_lens,
               int32_t N, int32_t K,
               const std::vector<ov::float16> &W_scale, int32_t W_group_size)
{
    auto ld_w = K/2, ld_in = K, ld_out = N;
    auto batch = input_offset_per_expert.size();
    std::cout << "Get reference_i4" << std::endl;
    auto expert_stride = ld_w * N;
    std::cout << "expert_stride : " << expert_stride << std::endl;
    int num_scale_groups = K / W_group_size;
    for (size_t b = 0; b < batch; b++) {
        int32_t expert_id = experts_ids[b];
        auto Wp = &W[expert_id * expert_stride];
        auto Inp = &In[input_offset_per_expert[b] * ld_in];
        auto Cp = &C[input_offset_per_expert[b] * ld_out];
        auto cur_m = input_tokens_lens[b];
        std::cout << "expert_id : " << expert_id << " cur_m : " << cur_m << std::endl;

        for (int j = 0; j < cur_m; j++) {
            for (int n = 0; n < N; n++) {
                auto W_r = Wp + n * ld_w;
                auto In_r = Inp + j * ld_in;
                float acc = 0.0f;
                int scale_group = 0;
                for (int k = 0; k < ld_w; k += W_group_size / 2) {
                    for (int ki = k; ki < k + W_group_size / 2; ++ki) {
                    // decompress
                        int8_t q0 = (int8_t)(W_r[ki] & 0x0F);
                        int8_t q1 = (int8_t)(W_r[ki] >> 4) & 0x0F;
                        if (q0 > 7)
                            q0 -= 16;
                        if (q1 > 7)
                            q1 -= 16;
                        float scale = float(W_scale[expert_id * N + n * num_scale_groups + scale_group]);
                        ov::float16 fa0 = ov::float16((float)q0 * (float)scale);
                        ov::float16 fa1 = ov::float16((float)q1 * (float)scale);
                        acc += (float)fa0 * In_r[2 * ki];
                        acc += (float)fa1 * In_r[2 * ki + 1];
    //                    std::cout << "ref_A[" << b << "][" << m << "][" << k * 2     << "] scale : " << scale << " zp : " << zp << " q: " << (int) q0 << " fa : " << fa0 << std::endl;
    //                    std::cout << "ref_A[" << b << "][" << m << "][" << k * 2 + 1 << "] scale : " << scale << " zp : " << zp << " q: " << (int) q1 << " fa : " << fa1 << std::endl;
                    }
                    scale_group++;
                }
                Cp[j * ld_out + n] = acc;
            }
        }
    }
}
#endif



TEST(moe_unit, moe_gemm_test_small_u4) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    size_t num_tokens = 10;
    size_t hidden_size = 2880;
    size_t num_total_experts = 32;
    size_t experts_out_N = 5760;
    int32_t num_active_experts_per_token = 2;
    size_t scale_group_size = hidden_size;
    size_t num_scale_groups = hidden_size / scale_group_size;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};
    {
        // weight to fill with 1.0f for initial test
        std::vector<uint8_t> experts_data_u4(num_total_experts * hidden_size * experts_out_N / 2);
        std::vector<ov::float16> scales_data(num_total_experts * num_scale_groups * experts_out_N);
        std::vector<ov::float16> zp_data(num_total_experts * num_scale_groups * experts_out_N);

        // create and quantize data
        auto experts_data_f16 = get_f16_weight(num_total_experts, experts_out_N, hidden_size);
        quantize_u4(experts_data_f16, experts_data_u4, num_total_experts, experts_out_N, hidden_size, hidden_size, scales_data, zp_data);

        auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        auto experts_layout = layout{experts_shape, data_types::u4, format::bfyx};
        auto experts_mem = engine.allocate_memory(experts_layout);
        set_values(experts_mem, experts_data_u4);

        auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        auto scale_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
        auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
        auto scale_mem = engine.allocate_memory(scale_layout);
        set_values(scale_mem, scales_data);

        auto zp_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
        auto zp_layout = layout{zp_shape, data_types::f16, format::bfyx};
        auto zp_mem = engine.allocate_memory(zp_layout);
        set_values(zp_mem, zp_data);

        topology topo_u4(input_layout("input", input_activation_layout),
                         data("moe_experts", experts_mem),
                         input_layout("experts_ids", experts_ids_layout),
                         input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                         input_layout("input_tokens_lens", input_tokens_lens_layout),
                         data("weight_scale", scale_mem),
                         data("weight_zp", zp_mem),
                         moe_gemm("moe_gemm",
                                  input_info("input"),
                                  input_info("moe_experts"),
                                  input_info("experts_ids"),
                                  input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                                  input_info("input_tokens_lens"),
                                  "",
                                  input_info("weight_scale"),
                                  input_info("weight_zp"),
                                  num_active_experts_per_token));

        std::vector<int32_t> input_tokens_lens = {5, 10, 5};
        auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
        auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens * num_active_experts_per_token), ov::Dimension(hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        std::vector<ov::float16> input_data(num_tokens * num_active_experts_per_token * hidden_size);
        size_t input_idx = 0;
        for (size_t l = 0; l < input_tokens_lens.size(); ++l) {
            for (size_t i = 0; i < input_tokens_lens[l] * hidden_size; ++i) {
                input_data[input_idx++] = (i % (l + 1)) / 10.0f + 1.0f;
            }
        }

        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data = {0, 5, 10};
        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data = {0, 5, 15};
        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network_u4(engine, topo_u4, config);
        network_u4.set_input_data("input", input_mem);
        network_u4.set_input_data("experts_ids", experts_ids_mem);
        network_u4.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network_u4.set_input_data("input_tokens_lens", input_tokens_lens_mem);

        auto outputs = network_u4.execute();
        auto output = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        std::cout << (float)output_ptr[0] << std::endl;

        std::cout << "U4 finished" << std::endl;
        //#############################################
        //    auto experts_shape_f16 = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
        //    auto experts_layout_f16 = layout{experts_shape, data_types::f16, format::bfyx};
        //    auto experts_mem_f16 = engine.allocate_memory(experts_layout_f16);
        //    set_values(experts_mem_f16, experts_data_f16);
        //
        //    topology topo_f16(
        //        input_layout("input", input_activation_layout),
        //        data("moe_experts", experts_mem_f16),
        //        input_layout("experts_ids", experts_ids_layout),
        //        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
        //        input_layout("input_tokens_lens", input_tokens_lens_layout),
        //        moe_gemm("moe_gemm", input_info("input"),
        //                             input_info("moe_experts"),
        //                             input_info("experts_ids"),
        //                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
        //                             input_info("input_tokens_lens"),
        //                             num_active_experts_per_token
        //        )
        //    );
        //    std::cout << "Run f16 network" << std::endl;
        //    network network_f16(engine, topo_f16, config);
        //    network_f16.set_input_data("input", input_mem);
        //    network_f16.set_input_data("experts_ids", experts_ids_mem);
        //    network_f16.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        //    network_f16.set_input_data("input_tokens_lens", input_tokens_lens_mem);
        //
        //    auto outputs_f16 = network_f16.execute();
        //    auto output_f16 = outputs_f16.begin()->second.get_memory();
        //    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_f16_ptr(output_f16, get_test_stream());

        //#############################################

        // ref f16
        //    auto out_ref_f16 = get_ref_moe_gemm(input_data, experts_data_f16, num_tokens, hidden_size, experts_out_N, experts_ids_data,
        //    input_offset_per_expert_data, input_tokens_lens,
        //               num_active_experts_per_token, true);
        // ref u4
        #if 1
        std::vector<ov::float16> out_ref_u4(num_tokens * num_active_experts_per_token * experts_out_N);
        reference_u4(experts_data_u4,
                     input_data,
                     out_ref_u4,
                     experts_ids_data,
                     input_offset_per_expert_data,
                     input_tokens_lens,
                     experts_out_N,
                     hidden_size,
                     scales_data,
                     zp_data,
                     scale_group_size);
        for (size_t m = 0; m < num_tokens * num_active_experts_per_token; m++) {
            for (size_t n = 0; n < experts_out_N; n++) {
                auto ref_u4 = out_ref_u4[m * experts_out_N + n];
                //            std::cout << "c[" << m << "][" << n << "] compute_u4: " << (float)output_ptr[m * experts_out_N + n]
                //                      << ", compute_f16 : " << (float)output_f16_ptr[m * experts_out_N + n] << ", ref_u4: " << ref_u4
                //                      << ", ref_f16:" << out_ref_f16[m * experts_out_N + n] << std::endl;
                auto tolerance = std::max(std::abs(ref_u4 * 0.01f), 0.1f);
                ASSERT_NEAR(output_ptr[m * experts_out_N + n], ref_u4, tolerance);
                if (std::abs(output_ptr[m * experts_out_N + n] - out_ref_u4[m * experts_out_N + n]) > tolerance) {
                    std::cout << "u4 !!! mismatch at [" << m << "][" << n << "]" << std::endl;
                }
            }
        }
        #endif
    }
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(),
                                                       ov::Dimension::dynamic(),
                                                       ov::Dimension(static_cast<int64_t>(hidden_size))};
        auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

        auto input = input_layout("input", input_activation_layout);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{ static_cast<int64_t>(num_total_experts), static_cast<int64_t>(experts_out_N), static_cast<int64_t>(hidden_size)}, data_types::u4, format::bfyx});
        auto weights_data = rg.generate_random_1d<uint8_t>(static_cast<int64_t>(num_total_experts) * static_cast<int64_t>(experts_out_N) * (hidden_size / 2), 0, 255);
        set_values(weights_mem, weights_data);
        auto w_prim = data("weights", weights_mem);

        auto scale_mem = engine.allocate_memory({ov::PartialShape{ static_cast<int64_t>(num_total_experts), static_cast<int64_t>(experts_out_N), 1}, data_types::f16, format::bfyx});
        auto scale_data = rg.generate_random_1d<ov::float16>(static_cast<int64_t>(num_total_experts) * static_cast<int64_t>(experts_out_N), -4.0f, 4.0f);
        set_values(scale_mem, scale_data);
        auto scale_prim = data("scale", scale_mem);

        auto zp_mem = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(num_total_experts), static_cast<int64_t>(experts_out_N), 1}, data_types::u8, format::bfyx});
        auto zp_data = rg.generate_random_1d<uint8_t>(num_total_experts * experts_out_N, 0, 255);
        set_values(zp_mem, zp_data);
        auto zp_prim = data("zp", zp_mem);
   
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "zp", data_types::f16, 3, 3);

        topology topology;
        topology.add(input);
        topology.add(w_prim);
        topology.add(scale_prim);
        topology.add(zp_prim);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{static_cast<int64_t>(num_total_experts), static_cast<int64_t>(num_tokens), static_cast<int64_t>(hidden_size)}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(static_cast<int64_t>(num_total_experts) * static_cast<int64_t>(num_tokens) * static_cast<int64_t>(hidden_size), -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
    }
}
TEST(moe_unit, moe_gemm_test_small_u4_generate_up) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    size_t num_tokens = 1;
    size_t hidden_size = 2880;
    size_t num_total_experts = 32;
    size_t experts_out_N = 5760;
    int32_t num_active_experts_per_token = 4;
    size_t scale_group_size = hidden_size;
    size_t num_scale_groups = hidden_size / scale_group_size;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};


    // weight to fill with 1.0f for initial test
    std::vector<uint8_t> experts_data_u4(num_total_experts * hidden_size * experts_out_N / 2);
    std::vector<ov::float16> scales_data(num_total_experts * num_scale_groups * experts_out_N);
    std::vector<ov::float16> zp_data(num_total_experts * num_scale_groups * experts_out_N);

    // create and quantize data
   // for (size_t e = 0; e < num_total_experts; ++e) {
   //     for (size_t n = 0; n < experts_out_N ; ++n) {
   //         for (size_t h = 0; h < hidden_size; ++h) {
   //             size_t idx = e * experts_out_N * hidden_size + n * hidden_size + h;
   //             experts_data_f16[idx] = static_cast<ov::float16>((e + (n % 3) + (h % 5) + 1) / 10.0f);
   //         }
   //     }
   // }
    auto experts_data_f16 = get_f16_weight(num_total_experts, experts_out_N, hidden_size);
    quantize_u4(experts_data_f16, experts_data_u4, num_total_experts, experts_out_N, hidden_size, hidden_size, scales_data, zp_data);

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::u4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    set_values(experts_mem, experts_data_u4);

    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    auto scale_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_layout);
    set_values(scale_mem, scales_data);

    auto zp_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
    auto zp_layout = layout{zp_shape, data_types::f16, format::bfyx};
    auto zp_mem = engine.allocate_memory(zp_layout);
    set_values(zp_mem, zp_data);

    auto moe_gemm_prim = moe_gemm("moe_gemm",
                             input_info("input"),
                             input_info("moe_experts"),
                             input_info("experts_ids"),
                             input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                             input_info("input_tokens_lens"),
                             "",
                             input_info("weight_scale"),
                             input_info("weight_zp"),
                             num_active_experts_per_token);
    moe_gemm_prim.has_bias = false;
    topology topo_u4(input_layout("input", input_activation_layout),
                     data("moe_experts", experts_mem),
                     input_layout("experts_ids", experts_ids_layout),
                     input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                     input_layout("input_tokens_lens", input_tokens_lens_layout),
                     data("weight_scale", scale_mem),
                     data("weight_zp", zp_mem),
                    moe_gemm_prim
    );

    std::vector<int32_t> input_tokens_lens = {1, 1};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data(num_tokens * hidden_size);
    for (size_t i = 0; i < input_tokens_lens[0] * hidden_size; ++i) {
        input_data[i] = (i % 5) / 10.0f;
    }
    set_values(input_mem, input_data);

    std::vector<int32_t> experts_ids_data = {0, 15, 20, 30};
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 0, 0, 0};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

//    auto experts_shape_f16 = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
//    auto experts_layout_f16 = layout{experts_shape, data_types::f16, format::bfyx};
//    auto experts_mem_f16 = engine.allocate_memory(experts_layout_f16);
//    set_values(experts_mem_f16, experts_data_f16);
//
//    topology topo_f16(
//        input_layout("input", input_activation_layout),
//        data("moe_experts", experts_mem_f16),
//        input_layout("experts_ids", experts_ids_layout),
//        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
//        input_layout("input_tokens_lens", input_tokens_lens_layout),
//        moe_gemm("moe_gemm", input_info("input"),
//                             input_info("moe_experts"),
//                             input_info("experts_ids"),
//                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
//                             input_info("input_tokens_lens"),
//                             num_active_experts_per_token
//        )
//    );
//    //#############################################
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
//    //#############################################
//    std::cout << "Set f16 network" << std::endl;
//    network network_f16(engine, topo_f16, config);
//    network_f16.set_input_data("input", input_mem);
//    network_f16.set_input_data("experts_ids", experts_ids_mem);
//    network_f16.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
//    network_f16.set_input_data("input_tokens_lens", input_tokens_lens_mem);

    //#############################################
    std::cout << "Set u4 network" << std::endl;
    network network_u4(engine, topo_u4, config);
    network_u4.set_input_data("input", input_mem);
    network_u4.set_input_data("experts_ids", experts_ids_mem);
    network_u4.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network_u4.set_input_data("input_tokens_lens", input_tokens_lens_mem);

    //#############################################
//    std::cout << "Run warm up" << std::endl;
//    for (size_t i = 0; i < 100; ++i) {
//        auto outputs_f16 = network_f16.execute();
//        auto output_f16 = outputs_f16.begin()->second.get_memory();
//        auto outputs_tmp = network_u4.execute();
//        auto output_tmp = outputs_tmp.begin()->second.get_memory();
//    }
    //#############################################
//    std::cout << "Run f16" << std::endl;
//    auto outputs_f16 = network_f16.execute();
//    auto output_f16 = outputs_f16.begin()->second.get_memory();
//    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_f16_ptr(output_f16, get_test_stream());

    std::cout << "Run u4" << std::endl;
    auto outputs = network_u4.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());

    //#############################################

    // ref f16
    auto out_ref_f16 = get_ref_moe_gemm(input_data, experts_data_f16, num_tokens, hidden_size, experts_out_N, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
               num_active_experts_per_token, true); 
    std::vector<ov::float16> out_ref_u4(num_tokens * num_active_experts_per_token * experts_out_N);
    // ref u4
    reference_u4(experts_data_u4, input_data, out_ref_u4, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
               experts_out_N, hidden_size, scales_data, zp_data, scale_group_size); 
    for (size_t m = 0; m < num_tokens * num_active_experts_per_token; m++) {
        for (size_t n = 0; n < experts_out_N; n++) {
            auto ref_u4 = out_ref_u4[m * experts_out_N + n];
//            std::cout << "c[" << m << "][" << n << "] compute_u4: " << (float)output_ptr[m * experts_out_N + n]
            //          << ", compute_f16 : " << (float)output_f16_ptr[m * experts_out_N + n] << ", ref_u4: " << ref_u4 
//                      << ", ref_f16:" << out_ref_f16[m * experts_out_N + n] << std::endl;
            auto tolerance = std::max(std::abs(ref_u4 * 0.01f), 0.1f); 
            ASSERT_NEAR(output_ptr[m * experts_out_N + n], ref_u4, tolerance);
//            if (std::abs(output_ptr[m * experts_out_N + n] - out_ref_u4[m * experts_out_N + n]) > 0.1f) {
//                std::cout << "!!! mismatch at [" << m << "][" << n << "]" << std::endl;
//            }
        }
    }

    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        int32_t num_tokens = 1;
        int32_t num_experts = 32;
        int32_t hidden_size = 2880;
        int32_t N = 5760;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(hidden_size)}, data_types::f16, format::bfyx};

        auto input = input_layout("input", input_activation_layout);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, hidden_size}, data_types::u4, format::bfyx});
        auto weights_data = rg.generate_random_1d<uint8_t>(num_experts * N * hidden_size / 2, 0, 255);
        set_values(weights_mem, weights_data);
        auto w_prim = data("weights", weights_mem);

        auto scale_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, 1}, data_types::f16, format::bfyx});
        auto scale_data = rg.generate_random_1d<ov::float16>(num_experts * N, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);
        auto scale_prim = data("scale", scale_mem);

        auto zp_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, 1}, data_types::i8, format::bfyx});
        auto zp_data = rg.generate_random_1d<int8_t>(num_experts * N, -4.0f, 4.0f);
        set_values(zp_mem, zp_data);
        auto zp_prim = data("zp", zp_mem);
    
        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "zp", data_types::f16, 3, 3);

        topology topology;
        topology.add(input);
        topology.add(w_prim);
        topology.add(scale_prim);
        topology.add(zp_prim);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, num_tokens, hidden_size}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * num_tokens * hidden_size, -1, 1);
        set_values(input_activation_data_mem, input_data);
        for (auto i = 0; i < 100; ++i) {
            network.set_input_data("input", input_activation_data_mem);
            auto output = network.execute().at("fc_prim").get_memory();
        }
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
    }

}

TEST(moe_unit, moe_gemm_test_small_i4_s32_first) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    size_t num_tokens = 10;
    size_t hidden_size = 2880;
    size_t num_total_experts = 32;
    size_t experts_out_N = 5760;
    int32_t num_active_experts_per_token = 4;
    size_t num_actually_used_experts = 8;
    size_t scale_group_size = 32;
    // [32, 32, 2/*num_scale_groups*/]
    size_t num_scale_groups = hidden_size / scale_group_size;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    std::vector<uint8_t>     experts_data_i4(num_total_experts * hidden_size * experts_out_N / 2);
    std::vector<ov::float16> scales_data(num_total_experts * experts_out_N * num_scale_groups);

    auto experts_data_f16 = get_f16_weight(num_total_experts, experts_out_N, hidden_size);
    quantize_i4_sym(experts_data_f16, experts_data_i4, num_total_experts, experts_out_N, hidden_size, scale_group_size, scales_data);

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::i4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    set_values(experts_mem, experts_data_i4);

    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    auto scale_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_layout);
    set_values(scale_mem, scales_data);

    auto moe_gemm_prim = moe_gemm("moe_gemm",
                             input_info("input"),
                             input_info("moe_experts"),
                             input_info("experts_ids"),
                             input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                             input_info("input_tokens_lens"),
                             "",
                             input_info("weight_scale"),
                             "",
                             num_active_experts_per_token);
    moe_gemm_prim.has_bias = false;
    topology topo_s4(input_layout("input", input_activation_layout),
                     data("moe_experts", experts_mem),
                     input_layout("experts_ids", experts_ids_layout),
                     input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                     input_layout("input_tokens_lens", input_tokens_lens_layout),
                     data("weight_scale", scale_mem),
                     moe_gemm_prim
    );
    // 10 * 4 experts
    std::vector<int32_t> input_tokens_lens = {5, 5, 5, 5, 5, 5, 5, 5};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_active_experts_per_token * num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data(num_active_experts_per_token * num_tokens * hidden_size);

    int offset = 0;
    for (size_t tl = 0; tl < num_actually_used_experts; ++tl) {
        std::cout << "token lens [" << tl << "] = " << input_tokens_lens[tl] << std::endl;
        for (size_t i = 0; i < input_tokens_lens[tl] * hidden_size; ++i) {
            input_data[offset] = (i % 5) / 10.0f;
            offset++;
        }
    }
    set_values(input_mem, input_data);

    std::vector<int32_t> experts_ids_data = {0, 2, 4, 6, 8, 10, 12, 14};

    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 5, 10, 15, 20, 25, 30, 35};

    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network_s4(engine, topo_s4, config);
    network_s4.set_input_data("input", input_mem);
    network_s4.set_input_data("experts_ids", experts_ids_mem);
    network_s4.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network_s4.set_input_data("input_tokens_lens", input_tokens_lens_mem);
    auto outputs = network_s4.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::cout << "out[0]: " << output_ptr[0] << std::endl;

    std::vector<float> out_ref_i4(num_tokens * num_active_experts_per_token * experts_out_N, 0.0f);
    std::cout << "I4 finished" << std::endl;
//    //#############################################
//    auto experts_shape_f16 = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
//    auto experts_layout_f16 = layout{experts_shape, data_types::f16, format::bfyx};
//    auto experts_mem_f16 = engine.allocate_memory(experts_layout_f16);
//    set_values(experts_mem_f16, experts_data_f16);
//
//    topology topo_f16(
//        input_layout("input", input_activation_layout),
//        data("moe_experts", experts_mem_f16),
//        input_layout("experts_ids", experts_ids_layout),
//        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
//        input_layout("input_tokens_lens", input_tokens_lens_layout),
//        moe_gemm("moe_gemm", input_info("input"),
//                             input_info("moe_experts"),
//                             input_info("experts_ids"),
//                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
//                             input_info("input_tokens_lens"),
//                             num_active_experts_per_token
//        )
//    );
//    std::cout << "Run f16 network" << std::endl;
//    network network_f16(engine, topo_f16, config);
//    network_f16.set_input_data("input", input_mem);
//    network_f16.set_input_data("experts_ids", experts_ids_mem);
//    network_f16.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
//    network_f16.set_input_data("input_tokens_lens", input_tokens_lens_mem);
//
//    auto outputs_f16 = network_f16.execute();
//    auto output_f16 = outputs_f16.begin()->second.get_memory();
//    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_f16_ptr(output_f16, get_test_stream());
//
//    //#############################################
//
//
//
//    // ref f16
//    auto out_ref_f16 = get_ref_moe_gemm(input_data, experts_data_f16, num_tokens, hidden_size, experts_out_N, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
//               num_active_experts_per_token, true); 
    // ref i4
    std::cout << "Calculate reference i4" << std::endl;
    #if 1
    reference_i4(experts_data_i4, input_data, out_ref_i4, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
               experts_out_N, hidden_size, scales_data, scale_group_size); 
    std::cout << "ref : " << out_ref_i4[0] << std::endl;
    int count = 0;
    for (size_t m = 0; m < num_tokens; m++) {
        for (size_t n = 0; n < experts_out_N; n++) {
            auto ref_i4 = out_ref_i4[m * experts_out_N + n];
//            std::cout << "c[" << m << "][" << n << "] compute_u4: " << (float)output_ptr[m * experts_out_N + n]
////                      << ", compute_f16 : " << (float)output_f16_ptr[m * experts_out_N + n]
//                      << ", ref_u4: " << ref_i4 
////                      << ", ref_f16:" << out_ref_f16[m * experts_out_N + n] << std::endl;
//                      << std::endl;
            auto tolerance = std::max(std::abs(ref_i4 * 0.01f), 0.1f); 
            ASSERT_NEAR(output_ptr[m * experts_out_N + n], ref_i4, tolerance);
            if (std::abs(output_ptr[m * experts_out_N + n] - out_ref_i4[m * experts_out_N + n]) > tolerance) {
//                std::cout << "!!! mismatch at [" << m << "][" << n << "]" << std::endl;
                count++;
            }
        }
    }
    std::cout << "fail count: " << count << std::endl;;
    #endif
#if 0
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        long int num_experts = static_cast<long int>(num_total_experts);
        long int M = static_cast<long int>(num_tokens);
        long int N = static_cast<long int>(experts_out_N);
        long int K = static_cast<long int>(hidden_size);
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(K)}, data_types::f16, format::bfyx};

        auto input = input_layout("input", input_activation_layout);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, K}, data_types::i4, format::bfyx});
        auto weights_data = rg.generate_random_1d<uint8_t>(num_experts * N * K / 2, 0, 255);
        set_values(weights_mem, weights_data);
        auto w_prim = data("weights", weights_mem);

        auto scale_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, static_cast<long int>(num_scale_groups)}, data_types::f16, format::bfyx});
        auto scale_data = rg.generate_random_1d<ov::float16>(num_experts * N * num_scale_groups, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);
        auto scale_prim = data("scale", scale_mem);

        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 3);

        topology topology;
        topology.add(input);
        topology.add(w_prim);
        topology.add(scale_prim);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, M, K}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * M * K, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
        // performance
        int to_run = (getenv("ITER") != nullptr) ? atoi(getenv("ITER")) : 0;
        for (int i = 0; i < to_run; ++i) {
            auto outputs_moe = network_s4.execute();
            auto output_moe = outputs_moe.begin()->second.get_memory();
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_moe_ptr(output_moe, get_test_stream());
            std::cout << "out[0]: " << output_moe_ptr[0] << std::endl;
            auto output_onednn = network.execute().at("fc_prim").get_memory();
//            std::cout << "output : " << output_onednn->get_layout().to_string() << std::endl;
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_onednn_ptr(output_onednn, get_test_stream());
            std::cout << "out_onednn[0]: " << output_onednn_ptr[0] << std::endl;
        }
    }

#endif
}

TEST(moe_unit, moe_gemm_test_small_i4_s32_generate_up) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    size_t num_tokens = 1;
    size_t hidden_size = 2880;
    size_t num_total_experts = 32;
    size_t experts_out_N = 5760;
    int32_t num_active_experts_per_token = 4;
    //size_t num_actually_used_experts = 4;
    size_t scale_group_size = 32;
    
    // [32, 32, 2/*num_scale_groups*/]
    size_t num_scale_groups = hidden_size / scale_group_size;

    auto input_activation_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(hidden_size)};
    auto input_activation_layout = layout{input_activation_shape, data_types::f16, format::bfyx};

    auto experts_data_f16 = get_f16_weight(num_total_experts, experts_out_N, hidden_size);
    std::vector<uint8_t>     experts_data_i4(num_total_experts * hidden_size * experts_out_N / 2);
    // [32, 32, 2/*num_scale_groups*/]
    std::vector<ov::float16> scales_data(num_total_experts * experts_out_N * num_scale_groups);

    // create and quantize data
    quantize_i4_sym(experts_data_f16, experts_data_i4, num_total_experts, experts_out_N, hidden_size, scale_group_size, scales_data);

    auto experts_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
    auto experts_layout = layout{experts_shape, data_types::i4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    set_values(experts_mem, experts_data_i4);

    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    auto scale_shape = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size / scale_group_size)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_layout);
    set_values(scale_mem, scales_data);

    auto moe_gemm_prim = moe_gemm("moe_gemm",
                             input_info("input"),
                             input_info("moe_experts"),
                             input_info("experts_ids"),
                             input_info("input_offset_per_expert"),  // this input will be croped to be same length as the actual used experts
                             input_info("input_tokens_lens"),
                             "",
                             input_info("weight_scale"),
                             "",
                             num_active_experts_per_token);
    moe_gemm_prim.has_bias = false;
    topology topo_s4(input_layout("input", input_activation_layout),
                     data("moe_experts", experts_mem),
                     input_layout("experts_ids", experts_ids_layout),
                     input_layout("input_offset_per_expert", input_offset_per_expert_layout),
                     input_layout("input_tokens_lens", input_tokens_lens_layout),
                     data("weight_scale", scale_mem),
                     moe_gemm_prim
    );

    std::vector<int32_t> input_tokens_lens = {1, 1, 1, 1};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    std::vector<ov::float16> input_data(num_tokens * hidden_size);

    size_t input_idx = 0;
//    for (size_t l = 0; l < input_tokens_lens.size(); ++l) {
    for (size_t l = 0; l < 1; ++l) {
        for (size_t i = 0; i < input_tokens_lens[l] * hidden_size; ++i) {
            input_data[input_idx++] = (i % (l + 1)) / 10.0f + 1.0f;
        }
    }

    set_values(input_mem, input_data);

    std::vector<int32_t> experts_ids_data = {1, 4, 7, 10};
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 0, 0, 0};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network_s4(engine, topo_s4, config);
    network_s4.set_input_data("input", input_mem);
    network_s4.set_input_data("experts_ids", experts_ids_mem);
    network_s4.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network_s4.set_input_data("input_tokens_lens", input_tokens_lens_mem);
    auto outputs = network_s4.execute();
    auto output = outputs.begin()->second.get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::cout << "out[0]: " << output_ptr[0] << std::endl;

    std::vector<float> out_ref_i4(num_tokens * num_active_experts_per_token * experts_out_N, 0.0f);
    std::cout << "I4 finished" << std::endl;
//    //#############################################
//    auto experts_shape_f16 = ov::PartialShape{ov::Dimension(num_total_experts), ov::Dimension(experts_out_N), ov::Dimension(hidden_size)};
//    auto experts_layout_f16 = layout{experts_shape, data_types::f16, format::bfyx};
//    auto experts_mem_f16 = engine.allocate_memory(experts_layout_f16);
//    set_values(experts_mem_f16, experts_data_f16);
//
//    topology topo_f16(
//        input_layout("input", input_activation_layout),
//        data("moe_experts", experts_mem_f16),
//        input_layout("experts_ids", experts_ids_layout),
//        input_layout("input_offset_per_expert", input_offset_per_expert_layout),
//        input_layout("input_tokens_lens", input_tokens_lens_layout),
//        moe_gemm("moe_gemm", input_info("input"),
//                             input_info("moe_experts"),
//                             input_info("experts_ids"),
//                             input_info("input_offset_per_expert"), // this input will be croped to be same length as the actual used experts
//                             input_info("input_tokens_lens"),
//                             num_active_experts_per_token
//        )
//    );
//    std::cout << "Run f16 network" << std::endl;
//    network network_f16(engine, topo_f16, config);
//    network_f16.set_input_data("input", input_mem);
//    network_f16.set_input_data("experts_ids", experts_ids_mem);
//    network_f16.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
//    network_f16.set_input_data("input_tokens_lens", input_tokens_lens_mem);
//
//    auto outputs_f16 = network_f16.execute();
//    auto output_f16 = outputs_f16.begin()->second.get_memory();
//    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_f16_ptr(output_f16, get_test_stream());
//
//    //#############################################
//
//
//
//    // ref f16
//    auto out_ref_f16 = get_ref_moe_gemm(input_data, experts_data_f16, num_tokens, hidden_size, experts_out_N, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
//               num_active_experts_per_token, true); 
    // ref i4
    std::cout << "Calculate reference i4" << std::endl;
    #if 1
    reference_i4(experts_data_i4, input_data, out_ref_i4, experts_ids_data, input_offset_per_expert_data, input_tokens_lens,
               experts_out_N, hidden_size, scales_data, scale_group_size); 
    std::cout << "ref : " << out_ref_i4[0] << std::endl;
    int count = 0;
    for (size_t m = 0; m < num_tokens * num_active_experts_per_token; m++) {
        for (size_t n = 0; n < experts_out_N; n++) {
            auto ref_i4 = out_ref_i4[m * experts_out_N + n];
//            std::cout << "c[" << m << "][" << n << "] compute_u4: " << (float)output_ptr[m * experts_out_N + n]
////                      << ", compute_f16 : " << (float)output_f16_ptr[m * experts_out_N + n]
//                      << ", ref_u4: " << ref_i4 
////                      << ", ref_f16:" << out_ref_f16[m * experts_out_N + n] << std::endl;
//                      << std::endl;
            auto tolerance = std::max(std::abs(ref_i4 * 0.01f), 0.1f); 
            ASSERT_NEAR(output_ptr[m * experts_out_N + n], ref_i4, tolerance);
            if (std::abs(output_ptr[m * experts_out_N + n] - out_ref_i4[m * experts_out_N + n]) > tolerance) {
                std::cout << __LINE__ << "!!! mismatch at [" << m << "][" << n << "]" << std::endl;
                count++;
            }
        }
    }
    std::cout << "fail count: " << count << std::endl;;
    #endif
#if 0
    {
        // run full experts
        std::cout << "Run onednn prim for full batch" << std::endl;
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        // Change input data of fully-connected node from bx to bf
        long int num_experts = static_cast<long int>(num_total_experts);
        long int M = static_cast<long int>(num_tokens);
        long int N = static_cast<long int>(experts_out_N);
        long int K = static_cast<long int>(hidden_size);
        auto input_activation_layout = layout{ov::PartialShape{num_experts, ov::Dimension::dynamic(), ov::Dimension(K)}, data_types::f16, format::bfyx};

        auto input = input_layout("input", input_activation_layout);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, K}, data_types::i4, format::bfyx});
        auto weights_data = rg.generate_random_1d<uint8_t>(num_experts * N * K / 2, 0, 255);
        set_values(weights_mem, weights_data);
        auto w_prim = data("weights", weights_mem);

        auto scale_mem = engine.allocate_memory({ov::PartialShape{ num_experts, N, static_cast<long int>(num_scale_groups)}, data_types::f16, format::bfyx});
        auto scale_data = rg.generate_random_1d<ov::float16>(num_experts * N * num_scale_groups, -4.0f, 4.0f);
        set_values(scale_mem, scale_data);
        auto scale_prim = data("scale", scale_mem);

        auto fc = fully_connected("fc_prim", input_info("input"), "weights", "", "scale", "", data_types::f16, 3, 3);

        topology topology;
        topology.add(input);
        topology.add(w_prim);
        topology.add(scale_prim);
        topology.add(fc);

        ov::intel_gpu::ImplementationDesc fc_impl = {format::bfyx, "", impl_types::onednn};
        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"fc_prim", fc_impl}}));
        network network(engine, topology, cfg);

        auto input_activation_data_layout = layout{ov::PartialShape{num_experts, M, K}, data_types::f16, format::bfyx}; 
        auto input_activation_data_mem = engine.allocate_memory(input_activation_data_layout);
        std::vector<ov::float16> input_data = rg.generate_random_1d<ov::float16>(num_experts * M * K, -1, 1);
        set_values(input_activation_data_mem, input_data);
        network.set_input_data("input", input_activation_data_mem);
        auto output = network.execute().at("fc_prim").get_memory();
        std::cout << "output : " << output->get_layout().to_string() << std::endl;
        // performance
        int to_run = (getenv("ITER") != nullptr) ? atoi(getenv("ITER")) : 0;
        for (int i = 0; i < to_run; ++i) {
            auto outputs_moe = network_s4.execute();
            auto output_moe = outputs_moe.begin()->second.get_memory();
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_moe_ptr(output_moe, get_test_stream());
            std::cout << "out[0]: " << output_moe_ptr[0] << std::endl;
            auto output_onednn = network.execute().at("fc_prim").get_memory();
//            std::cout << "output : " << output_onednn->get_layout().to_string() << std::endl;
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_onednn_ptr(output_onednn, get_test_stream());
            std::cout << "out_onednn[0]: " << output_onednn_ptr[0] << std::endl;
        }
    }

#endif
}
