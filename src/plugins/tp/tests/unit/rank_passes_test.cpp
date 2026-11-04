// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "l1_test_utils.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/paged_attention.hpp"
#include "ov_ops/tp_collective.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

/// The number of output features the projection's weight still carries.
int64_t lm_head_rows(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op);
        if (!matmul || matmul->get_friendly_name() != "lm_head")
            continue;
        const auto& weight = matmul->input_value(1).get_partial_shape();
        return weight[matmul->get_transpose_b() ? weight.size() - 2 : weight.size() - 1].get_length();
    }
    return -1;
}

/// KV heads rank `rank` owns: whole heads, remainder spread over the first ranks.
int64_t local_kv_heads(const BlockConfig& config, size_t rank, size_t world_size) {
    const auto base = config.num_kv_heads / world_size;
    const auto remainder = config.num_kv_heads % world_size;
    return static_cast<int64_t>(base + (rank < remainder ? 1 : 0));
}

std::shared_ptr<ov::Node> paged_attention(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> found;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
            EXPECT_EQ(found, nullptr) << "more than one PagedAttention";
            found = op;
        }
    }
    return found;
}

const auto& all_reduce = ov::op::internal::TPAllReduce::get_type_info_static();
const auto& gather = ov::op::internal::TPGather::get_type_info_static();
const auto& slice = ov::op::v8::Slice::get_type_info_static();

}  // namespace

class TPRankPassesTest : public ::testing::TestWithParam<size_t> {};

TEST_P(TPRankPassesTest, GivesEveryRankOneBandOfTheVocabulary) {
    const size_t world_size = GetParam();
    const size_t vocab = 1536;  // divides by two, three and four
    for (const auto& model : rank_models(make_block_with_lm_head(BlockConfig{}, vocab), world_size)) {
        EXPECT_EQ(lm_head_rows(model), static_cast<int64_t>(vocab / world_size));
        EXPECT_EQ(count_ops_of_type(model, gather), 1u);
        // The band is cut out of the stored weight: a Slice left in the graph is folded at every compile.
        EXPECT_EQ(count_ops_of_type(model, slice), 0u);
    }
}

TEST_P(TPRankPassesTest, OnlyRankZeroSeesTheWholeVocabulary) {
    const size_t world_size = GetParam();
    const size_t vocab = 1536;
    const auto models = rank_models(make_block_with_lm_head(BlockConfig{}, vocab), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto& shape = models[rank]->get_results()[0]->get_output_partial_shape(0);
        const auto expected = static_cast<int64_t>(rank == 0 ? vocab : vocab / world_size);
        EXPECT_EQ(shape[shape.size() - 1].get_length(), expected) << "rank " << rank;
    }
}

TEST_P(TPRankPassesTest, KeepsTheVocabularyProjectionWholeOnRequest) {
    const size_t world_size = GetParam();
    const size_t vocab = 1536;
    for (const auto& model : rank_models(make_block_with_lm_head(BlockConfig{}, vocab), world_size, false)) {
        EXPECT_EQ(lm_head_rows(model), static_cast<int64_t>(vocab));
        EXPECT_EQ(count_ops_of_type(model, gather), 0u);
    }
}

TEST_P(TPRankPassesTest, InsertsNoGatherWithoutAVocabularyProjection) {
    for (const auto& model : rank_models(make_transformer_block(BlockConfig{}), GetParam()))
        EXPECT_EQ(count_ops_of_type(model, gather), 0u);
}

TEST_P(TPRankPassesTest, CutsEveryWeightAtCompileTime) {
    // The block's intermediate size does not divide by 3 or 4, so the split falls back on whole
    // quantization groups -- still without a runtime Slice.
    auto model = make_transformer_block(BlockConfig{});
    ASSERT_EQ(count_ops_of_type(model, slice), 0u);
    for (const auto& rank_model : rank_models(model, GetParam()))
        EXPECT_EQ(count_ops_of_type(rank_model, slice), 0u);
}

TEST_P(TPRankPassesTest, InsertsOneAllReducePerRowParallelProjection) {
    BlockConfig dense;
    dense.gated_mlp = false;
    for (const auto& config : {BlockConfig{}, dense}) {
        for (const auto& model : rank_models(make_transformer_block(config), GetParam()))
            EXPECT_EQ(count_ops_of_type(model, all_reduce), 2u);
    }
    for (const auto& model : rank_models(make_paged_attention_block(BlockConfig{}), GetParam()))
        EXPECT_EQ(count_ops_of_type(model, all_reduce), 2u);
}

TEST_P(TPRankPassesTest, LocalizesTheKvCacheVariables) {
    const size_t world_size = GetParam();
    BlockConfig config;
    config.stateful = true;
    const auto models = rank_models(make_transformer_block(config), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto variables = models[rank]->get_variables();
        ASSERT_EQ(variables.size(), 2u);
        for (const auto& variable : variables) {
            EXPECT_EQ(variable->get_info().data_shape[1].get_length(), local_kv_heads(config, rank, world_size))
                << variable->get_info().variable_id << " rank " << rank;
        }
    }
}

TEST_P(TPRankPassesTest, LocalizesEveryKvCacheInitializer) {
    const size_t world_size = GetParam();
    BlockConfig config;
    config.stateful = true;
    const auto models = rank_models(make_transformer_block(config), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        // Every ReadValue carries its own initializer chain.
        size_t checked = 0;
        for (const auto& op : models[rank]->get_ordered_ops()) {
            auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
            if (!read_value || read_value->get_input_size() == 0)
                continue;
            EXPECT_EQ(read_value->get_output_partial_shape(0)[1].get_length(), local_kv_heads(config, rank, world_size));
            ++checked;
        }
        EXPECT_EQ(checked, 2u) << "expected one initializer per KV cache";
    }
}

TEST_P(TPRankPassesTest, LocalizesThePagedAttentionHeadCounts) {
    const size_t world_size = GetParam();
    BlockConfig config;
    const auto models = rank_models(make_paged_attention_block(config), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto attention = paged_attention(models[rank]);
        ASSERT_NE(attention, nullptr);
        // The GPU plugin reads the KV head count from here; the cache sizing pass reads the value entry.
        const auto& rt_info = attention->get_rt_info();
        for (const char* key : {"num_k_heads", "num_v_heads"}) {
            const auto entry = rt_info.find(key);
            ASSERT_NE(entry, rt_info.end()) << key;
            EXPECT_EQ(entry->second.as<int64_t>(), local_kv_heads(config, rank, world_size)) << key << " rank " << rank;
        }
        // Head size is per head and does not split.
        EXPECT_EQ(rt_info.at("k_head_size").as<int64_t>(), static_cast<int64_t>(config.head_dim));
        EXPECT_EQ(rt_info.at("v_head_size").as<int64_t>(), static_cast<int64_t>(config.head_dim));
    }
}

TEST_P(TPRankPassesTest, LocalizesThePagedAttentionOperands) {
    const size_t world_size = GetParam();
    BlockConfig config;
    const auto models = rank_models(make_paged_attention_block(config), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto attention = paged_attention(models[rank]);
        ASSERT_NE(attention, nullptr);
        const int64_t kv = local_kv_heads(config, rank, world_size);
        const int64_t q = kv * static_cast<int64_t>(config.num_heads / config.num_kv_heads);
        const auto head_dim = static_cast<int64_t>(config.head_dim);
        // Nothing in the graph names a head count, so the shapes prove the split landed.
        EXPECT_EQ(attention->get_input_partial_shape(0)[1].get_length(), q * head_dim) << "rank " << rank;
        EXPECT_EQ(attention->get_input_partial_shape(1)[1].get_length(), kv * head_dim) << "rank " << rank;
        EXPECT_EQ(attention->get_input_partial_shape(2)[1].get_length(), kv * head_dim) << "rank " << rank;
        EXPECT_EQ(attention->get_output_partial_shape(0)[1].get_length(), q * head_dim) << "rank " << rank;
    }
}

TEST_P(TPRankPassesTest, LeavesThePagedAttentionCachePortsToTheRuntime) {
    // The plugin that allocates the cache decides its shape and precision.
    for (const auto& model : rank_models(make_paged_attention_block(BlockConfig{}), GetParam())) {
        size_t cache_ports = 0;
        for (const auto& parameter : model->get_parameters()) {
            const auto& name = parameter->get_friendly_name();
            if (name.rfind("key_cache.", 0) != 0 && name.rfind("value_cache.", 0) != 0)
                continue;
            ++cache_ports;
            EXPECT_TRUE(parameter->get_element_type().is_dynamic()) << name;
            EXPECT_TRUE(parameter->get_output_partial_shape(0).is_dynamic()) << name;
        }
        EXPECT_EQ(cache_ports, 2u);
    }
}

TEST_P(TPRankPassesTest, LeavesNoAnnotations) {
    for (const auto& model : rank_models(make_block_with_lm_head(BlockConfig{}, 1536), GetParam())) {
        for (const auto& node : model->get_ordered_ops()) {
            EXPECT_EQ(get_shard(*node), nullptr) << node->get_friendly_name();
            EXPECT_EQ(get_role(*node), nullptr) << node->get_friendly_name();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TPRankPasses,
                         TPRankPassesTest,
                         ::testing::Values(2u, 3u, 4u),
                         [](const ::testing::TestParamInfo<size_t>& info) {
                             return "world" + std::to_string(info.param);
                         });

}  // namespace ov::tp::tests
