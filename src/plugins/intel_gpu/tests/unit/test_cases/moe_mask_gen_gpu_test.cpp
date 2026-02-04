// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <numeric>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_mask_gen.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
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