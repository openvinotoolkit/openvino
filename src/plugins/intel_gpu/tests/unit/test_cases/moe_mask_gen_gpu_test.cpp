// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
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
    int64_t num_total_experts = 32;
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
    std::vector<float> experts_info_start_idx_ref = {0, 0, 0, 0, 30, 30, 30, 30, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60};
    const auto& output_experts_info_start_idx = outputs.at("experts_info_start_idx").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_experts_info_start_idx_ptr(output_experts_info_start_idx, get_test_stream());
    for (size_t i = 0; i < static_cast<size_t>(num_total_experts); i++) {
        ASSERT_EQ(static_cast<int32_t>(output_experts_info_start_idx_ptr[i]), experts_info_start_idx_ref[i]);
    }
    std::vector<float> tokens_lens_per_expert_ref = {30, 30};
    const auto& output_tokens_lens_per_expert = outputs.at("tokens_lens_per_expert").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_tokens_lens_per_expert_ptr(output_tokens_lens_per_expert, get_test_stream());
    for (size_t i = 0; i < static_cast<size_t>(num_actually_used_experts); i++) {
        ASSERT_EQ(static_cast<int32_t>(output_tokens_lens_per_expert_ptr[i]), tokens_lens_per_expert_ref[i]);
    }
}

namespace {

// Distinct experts per token, as produced by a router TopK
std::vector<int32_t> make_router_ids(int32_t num_tokens, int32_t top_k, int32_t num_experts, std::mt19937& gen) {
    std::vector<int32_t> topk_idx(static_cast<size_t>(num_tokens) * top_k);
    std::vector<int32_t> experts(num_experts);
    std::iota(experts.begin(), experts.end(), 0);
    for (int32_t t = 0; t < num_tokens; t++) {
        std::shuffle(experts.begin(), experts.end(), gen);
        std::copy_n(experts.begin(), top_k, topk_idx.begin() + t * top_k);
    }
    return topk_idx;
}

// Stable counting sort of the (token, slot) rows by expert
struct SortReference {
    std::vector<int32_t> end_offsets;  // inclusive end row of each expert
    std::vector<int32_t> token_index;  // token of each sorted row
    std::vector<int32_t> row_lut;      // sorted row of each (token, slot) position
};

SortReference make_sort_reference(const std::vector<int32_t>& topk_idx, int32_t top_k, int32_t num_experts) {
    SortReference ref;
    std::vector<int32_t> lens(num_experts, 0);
    for (auto e : topk_idx) {
        lens[e]++;
    }
    ref.end_offsets.resize(num_experts);
    std::partial_sum(lens.begin(), lens.end(), ref.end_offsets.begin());
    std::vector<int32_t> next_row(num_experts, 0);
    for (int32_t e = 1; e < num_experts; e++) {
        next_row[e] = ref.end_offsets[e - 1];
    }
    ref.token_index.resize(topk_idx.size());
    ref.row_lut.resize(topk_idx.size());
    for (size_t s = 0; s < topk_idx.size(); s++) {
        const int32_t row = next_row[topk_idx[s]]++;
        ref.token_index[row] = static_cast<int32_t>(s) / top_k;
        ref.row_lut[s] = row;
    }
    return ref;
}

}  // namespace

// GEMM3_SWIGLU outputs: counting sort of the (token, slot) pairs by expert, stable by flat position.
// With the dynamic input the token counts run through one network, with the static input a network is built per token count.
class moe_mask_gen_3gemm : public ::testing::TestWithParam<std::tuple<int32_t, int32_t, bool>> {};  // top_k, experts, dynamic input

TEST_P(moe_mask_gen_3gemm, matches_reference) {
    auto& engine = get_test_engine();
    const int32_t top_k = std::get<0>(GetParam());
    const int32_t num_experts = std::get<1>(GetParam());
    const bool dynamic_input = std::get<2>(GetParam());

    const std::vector<std::string> names{"token_index", "end_offsets", "row_lut"};
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto build_network = [&](const ov::PartialShape& topk_shape) {
        topology topology(input_layout("topk", layout{topk_shape, data_types::i32, format::bfyx}),
                          moe_mask_gen("mask_gen", input_info("topk"), num_experts, top_k, moe_mask_gen::expert_type::GEMM3_SWIGLU));
        for (size_t i = 0; i < names.size(); i++) {
            topology.add(reorder(names[i], input_info("mask_gen", static_cast<int>(i)), format::bfyx, data_types::i32));
        }
        return get_network(engine, topology, config, get_test_stream_ptr(), false);
    };
    auto dynamic_network = dynamic_input ? build_network(ov::PartialShape{ov::Dimension::dynamic(), top_k}) : nullptr;

    std::mt19937 gen((top_k * 31) + num_experts);
    for (const int32_t num_tokens : {1, 38, 256, 100, 1, 731}) {
        const auto topk_idx = make_router_ids(num_tokens, top_k, num_experts, gen);
        const auto ref = make_sort_reference(topk_idx, top_k, num_experts);

        auto network = dynamic_input ? dynamic_network : build_network(ov::PartialShape{num_tokens, top_k});
        auto topk_mem = engine.allocate_memory(layout{ov::PartialShape{num_tokens, top_k}, data_types::i32, format::bfyx});
        set_values(topk_mem, topk_idx);
        network->set_input_data("topk", topk_mem);
        auto outputs = network->execute();

        auto check = [&](size_t output_idx, const std::vector<int32_t>& expected) {
            auto mem = outputs.at(names[output_idx]).get_memory();
            cldnn::mem_lock<int32_t, mem_lock_type::read> ptr(mem, get_test_stream());
            for (size_t i = 0; i < expected.size(); i++) {
                ASSERT_EQ(ptr[i], expected[i]) << names[output_idx] << " index " << i << " tokens " << num_tokens;
            }
        };
        check(moe_mask_gen::MoEMaskGen3GemmOutputIdx::TOKEN_INDEX, ref.token_index);
        check(moe_mask_gen::MoEMaskGen3GemmOutputIdx::EXPERT_END_OFFSETS, ref.end_offsets);
        check(moe_mask_gen::MoEMaskGen3GemmOutputIdx::ROW_LUT, ref.row_lut);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         moe_mask_gen_3gemm,
                         ::testing::Values(std::make_tuple(6, 64, true),
                                           std::make_tuple(6, 64, false),
                                           std::make_tuple(8, 56, true),
                                           std::make_tuple(8, 56, false),
                                           std::make_tuple(8, 256, true),
                                           std::make_tuple(8, 256, false)));
