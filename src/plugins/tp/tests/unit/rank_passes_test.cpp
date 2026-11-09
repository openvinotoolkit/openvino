// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
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

TEST_P(TPRankPassesTest, CountsHowItCutTheWeights) {
    auto analyzed = make_transformer_block(BlockConfig{});
    analyze(analyzed, GetParam());
    auto model = analyzed->clone();
    auto stats = std::make_shared<TPShardingStats>();
    TPWeightShardingPass(0, GetParam(), stats).run_on_model(model);
    // Column-parallel weights are cut along their outermost axis, row-parallel ones are not.
    EXPECT_GT(stats->views, 0u);
    EXPECT_GT(stats->copies, 0u);
    EXPECT_GT(stats->copied_bytes, 0u);
    EXPECT_EQ(stats->runtime_slices, 0u);
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

// Packed 4-bit elements keep their values when a rank's range starts or ends inside a byte, along
// the outer axis (rows of three) as well as the inner one.
TEST(TPRankPassesNibbleTest, CutsPacked4BitElementsInsideAByte) {
    for (const int64_t axis : {0, 1}) {
        SCOPED_TRACE("axis " + std::to_string(axis));
        const auto shape = axis == 0 ? ov::Shape{5, 3} : ov::Shape{3, 5};
        std::vector<int> values(15);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<int>(i);
        const auto constant = ov::op::v0::Constant::create(ov::element::u4, shape, values);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{constant}, ov::ParameterVector{});
        const std::vector<Segment> segments{{5, 1}};
        set_shard(constant, ShardAnnotation(ShardStrategy::COLUMN_PARALLEL, axis, segments, "test"));
        for (size_t rank = 0; rank < 2; ++rank) {
            const auto rank_model = model->clone();
            TPWeightShardingPass(rank, 2).run_on_model(rank_model);
            const auto sliced = ov::as_type_ptr<ov::op::v0::Constant>(rank_model->get_results()[0]->get_input_node_shared_ptr(0));
            ASSERT_NE(sliced, nullptr);
            const auto range = rank_ranges(segments, rank, 2).front();
            std::vector<int> expected;
            for (size_t i = 0; i < values.size(); ++i) {
                const size_t index = axis == 0 ? i / shape[1] : i % shape[1];
                if (index >= range.offset && index < range.offset + range.length)
                    expected.push_back(values[i]);
            }
            EXPECT_EQ(sliced->cast_vector<int>(), expected) << "rank " << rank;
        }
    }
}

// Groups of 64: one per unit of two query heads, 7 in the MLP. The MLP's rows of 7 zero points, and
// ranks that end after an odd number of groups on three and four ranks, cut the packed u4 zero
// points inside a byte; every weight is still cut at compile time.
TEST(TPRankPassesNibbleTest, CutsZeroPointsInsideAByteAtCompileTime) {
    BlockConfig config;
    config.group_size = 64;
    for (const size_t world_size : {2u, 3u, 4u}) {
        SCOPED_TRACE("world " + std::to_string(world_size));
        auto analyzed = make_transformer_block(config);
        const auto report = analyze(analyzed, world_size);
        EXPECT_TRUE(report->runtime_slices.empty()) << report->runtime_slices.front();
        for (size_t rank = 0; rank < world_size; ++rank)
            EXPECT_EQ(count_ops_of_type(rank_model(analyzed, rank, world_size), slice), 0u) << "rank " << rank;
    }
}

// Units of two query heads (64 features) and groups of 128: two ranks end on a group boundary, but
// three (4, 2, 2 heads) and four (2 each) end at 192, inside a group of the O projection, which
// only a runtime Slice can cut.
TEST(TPRankPassesRuntimeSliceTest, SlicesOnlyTheOProjectionWhenARankEndsInsideAGroup) {
    BlockConfig config;
    config.group_size = 128;
    config.intermediate = 512;
    for (const size_t world_size : {2u, 3u, 4u}) {
        SCOPED_TRACE("world " + std::to_string(world_size));
        const size_t expected = world_size == 2 ? 0u : 1u;
        auto analyzed = make_transformer_block(config);
        const auto report = analyze(analyzed, world_size);
        EXPECT_EQ(report->runtime_slices.size(), expected);
        for (size_t rank = 0; rank < world_size; ++rank) {
            const auto model = rank_model(analyzed, rank, world_size);
            size_t sliced = 0;
            for (const auto& op : model->get_ordered_ops()) {
                if (!ov::is_type<ov::op::v8::Slice>(op))
                    continue;
                ++sliced;
                const auto consumer = op->output(0).get_target_inputs().begin()->get_node();
                ASSERT_TRUE(ov::is_type<ov::op::v0::MatMul>(consumer));
                // The O projection takes the attention output, not the MLP's gated product.
                EXPECT_FALSE(ov::is_type<ov::op::v1::Multiply>(consumer->get_input_node_ptr(0)));
                const auto& in = consumer->get_input_partial_shape(0);
                EXPECT_EQ(in[in.size() - 1].get_length(),
                          static_cast<int64_t>(rank_length({{config.num_heads * config.head_dim, 2 * config.head_dim}},
                                                           rank,
                                                           world_size)));
            }
            EXPECT_EQ(sliced, expected) << "rank " << rank;
        }
    }
}

// Fewer KV heads than ranks: every rank keeps the one KV head its query heads use.
struct FewKvHeads {
    size_t kv_heads;
    size_t world_size;
};

class TPRankPassesFewKvHeadsTest : public ::testing::TestWithParam<FewKvHeads> {
protected:
    static BlockConfig config() {
        BlockConfig c;
        c.num_heads = 12;  // divides by 2, 3 and 4
        c.num_kv_heads = GetParam().kv_heads;
        return c;
    }
};

TEST_P(TPRankPassesFewKvHeadsTest, ReportsTheRanksSharingAKvHead) {
    auto model = make_transformer_block(config());
    const auto report = analyze(model, GetParam().world_size);
    EXPECT_EQ(report->kv_replicas, GetParam().world_size / GetParam().kv_heads);
}

TEST_P(TPRankPassesFewKvHeadsTest, KeepsOneKvHeadAndAShareOfItsGroup) {
    const auto [kv_heads, world_size] = GetParam();
    auto c = config();
    c.stateful = true;
    const auto models = rank_models(make_transformer_block(c), world_size);
    const auto head_dim = static_cast<int64_t>(c.head_dim);
    for (size_t rank = 0; rank < world_size; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        for (const auto& variable : models[rank]->get_variables())
            EXPECT_EQ(variable->get_info().data_shape[1].get_length(), 1);
        for (const auto& op : models[rank]->get_ordered_ops()) {
            if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
                // The query heads of the rank, and its KV head expanded to as many.
                const auto local_q = static_cast<int64_t>(c.num_heads / world_size);
                for (size_t i = 0; i < 3; ++i)
                    EXPECT_EQ(sdpa->get_input_partial_shape(i)[1].get_length(), local_q) << "input " << i;
                EXPECT_EQ(sdpa->get_input_partial_shape(1)[3].get_length(), head_dim);
            }
        }
    }
}

TEST_P(TPRankPassesFewKvHeadsTest, LocalizesThePagedAttentionHeads) {
    const auto [kv_heads, world_size] = GetParam();
    auto c = config();
    const auto models = rank_models(make_paged_attention_block(c), world_size);
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto attention = paged_attention(models[rank]);
        ASSERT_NE(attention, nullptr);
        EXPECT_EQ(attention->get_rt_info().at("num_k_heads").as<int64_t>(), 1);
        EXPECT_EQ(attention->get_rt_info().at("num_v_heads").as<int64_t>(), 1);
        const auto head_dim = static_cast<int64_t>(c.head_dim);
        EXPECT_EQ(attention->get_input_partial_shape(0)[1].get_length(),
                  static_cast<int64_t>(c.num_heads / world_size) * head_dim);
        EXPECT_EQ(attention->get_input_partial_shape(1)[1].get_length(), head_dim);
    }
}

INSTANTIATE_TEST_SUITE_P(TPRankPasses,
                         TPRankPassesFewKvHeadsTest,
                         ::testing::Values(FewKvHeads{1, 2}, FewKvHeads{1, 3}, FewKvHeads{1, 4}, FewKvHeads{2, 4}),
                         [](const ::testing::TestParamInfo<FewKvHeads>& info) {
                             return "kv" + std::to_string(info.param.kv_heads) + "_world" + std::to_string(info.param.world_size);
                         });

// Fused qkv_proj / gate_up_proj: one MatMul whose output is split; every rank's weight holds its
// share of every part, and the split takes the rank's parts.
struct FusedCase {
    std::string name;
    BlockConfig config;
};

static std::vector<FusedCase> fused_cases() {
    std::vector<FusedCase> cases;
    for (const auto qkv : {Fused::VARIADIC_SPLIT, Fused::SLICES}) {
        for (const auto gate_up : {Fused::SPLIT, Fused::VARIADIC_SPLIT}) {
            BlockConfig c;
            c.qkv = qkv;
            c.gate_up = gate_up;
            cases.push_back({std::string(qkv == Fused::SLICES ? "slices" : "variadic") +
                                 (gate_up == Fused::SPLIT ? "_chunk" : "_variadic"),
                             c});
        }
    }
    BlockConfig c;
    c.qkv = Fused::SLICES;
    c.gate_up = Fused::SPLIT;
    c.with_bias = true;
    c.stateful = true;
    cases.push_back({"bias_stateful", c});
    c = BlockConfig{};
    c.qkv = Fused::SLICES;
    c.num_heads = 24;  // an even number of quantization groups per rank on 2, 3 and 4 ranks
    c.num_kv_heads = 1;
    cases.push_back({"one_kv_head", c});
    return cases;
}

class TPRankPassesFusedTest : public ::testing::TestWithParam<std::tuple<FusedCase, size_t>> {};

TEST_P(TPRankPassesFusedTest, GivesEveryRankItsShareOfEveryPart) {
    const auto& [fused, world_size] = GetParam();
    const auto& c = fused.config;
    auto analyzed = make_transformer_block(c);
    const auto report = analyze(analyzed, world_size);
    EXPECT_EQ(report->attention_blocks, 1u);
    EXPECT_EQ(report->mlp_blocks, 1u);
    EXPECT_TRUE(report->runtime_slices.empty()) << report->runtime_slices.front();
    EXPECT_EQ(roles(analyzed).size(), c.gate_up != Fused::NONE ? 4u : 5u);  // qkv, o, gate_up or gate + up, down

    const auto replicas = world_size > c.num_kv_heads ? world_size / c.num_kv_heads : 1;
    const auto group = c.num_heads / c.num_kv_heads;
    const auto hd = static_cast<int64_t>(c.head_dim);
    for (size_t rank = 0; rank < world_size; ++rank) {
        SCOPED_TRACE("rank " + std::to_string(rank));
        const auto model = rank_model(analyzed, rank, world_size);
        const auto q_heads = static_cast<int64_t>(rank_length({{c.num_heads, group / replicas}}, rank, world_size));
        const auto kv_heads = replicas > 1 ? 1 : static_cast<int64_t>(rank_length({{c.num_kv_heads, 1}}, rank, world_size));
        int64_t fused_qkv = -1, fused_gate_up = -1, down_in = -1;
        for (const auto& op : model->get_ordered_ops()) {
            const auto out = op->get_output_partial_shape(0);
            if (op->get_friendly_name() == "qkv_proj")
                fused_qkv = out[2].get_length();
            if (op->get_friendly_name() == "gate_up_proj")
                fused_gate_up = out[2].get_length();
            if (ov::is_type<ov::op::v0::MatMul>(op) && ov::is_type<ov::op::v1::Multiply>(op->get_input_node_ptr(0)))
                down_in = op->get_input_partial_shape(0)[2].get_length();
            if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op)) {
                for (size_t i = 0; i < 3; ++i)
                    EXPECT_EQ(sdpa->get_input_partial_shape(i)[1].get_length(), q_heads) << "input " << i;
            }
        }
        EXPECT_EQ(fused_qkv, (q_heads + 2 * kv_heads) * hd);
        if (c.gate_up != Fused::NONE) {
            EXPECT_EQ(fused_gate_up, 2 * down_in);
        }
        EXPECT_EQ(count_ops_of_type(model, all_reduce), 2u);
    }
}

INSTANTIATE_TEST_SUITE_P(TPRankPasses,
                         TPRankPassesFusedTest,
                         ::testing::Combine(::testing::ValuesIn(fused_cases()), ::testing::Values(2u, 3u, 4u)),
                         [](const ::testing::TestParamInfo<std::tuple<FusedCase, size_t>>& info) {
                             return std::get<0>(info.param).name + "_world" + std::to_string(std::get<1>(info.param));
                         });

TEST(TPRankPassesTest, RejectsKvHeadsThatDoNotDivideTheRanks) {
    BlockConfig c;
    c.num_heads = 12;
    c.num_kv_heads = 2;
    auto model = make_transformer_block(c);
    OV_EXPECT_THROW(analyze(model, 3), ov::Exception, ::testing::HasSubstr("world size 3 does not divide them evenly"));
}

}  // namespace ov::tp::tests
