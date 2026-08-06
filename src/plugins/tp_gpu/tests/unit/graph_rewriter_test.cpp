// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "graph_rewriter.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "tp_gpu/op/tp_all_reduce.hpp"
#include "tp_test_models.hpp"

namespace ov::tp_gpu::tests {
namespace {

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops())
        count += op->get_type_info().is_castable(type) ? 1 : 0;
    return count;
}

/// Runs the rewriter for every rank of a world and hands back the models.
std::vector<std::shared_ptr<ov::Model>> rewrite_all_ranks(const std::shared_ptr<ov::Model>& model,
                                                          const ShardingPlan& plan,
                                                          uint32_t world_size) {
    std::vector<std::shared_ptr<ov::Model>> per_rank;
    per_rank.reserve(world_size);
    for (uint32_t rank = 0; rank < world_size; ++rank)
        per_rank.push_back(GraphRewriter::rewrite(model, plan, rank, world_size));
    return per_rank;
}

}  // namespace

// ---------------------------------------------------------------------------
// analyze()
// ---------------------------------------------------------------------------

TEST(TPGraphRewriterAnalyze, FindsProjectionsAndGeometry) {
    BlockConfig config;
    auto plan = GraphRewriter::analyze(make_transformer_block(config));

    EXPECT_EQ(plan.num_layers, 1);
    EXPECT_EQ(plan.num_heads, static_cast<int>(config.num_heads));
    EXPECT_EQ(plan.num_kv_heads, static_cast<int>(config.num_kv_heads));
    EXPECT_EQ(plan.head_dim, static_cast<int>(config.head_dim));
    EXPECT_EQ(plan.intermediate_size, static_cast<int>(config.intermediate));

    // q, k, v, o, gate, up, down -- and nothing else.
    EXPECT_EQ(plan.linears.size(), 7u);
    EXPECT_EQ(GraphRewriter::count_collectives(plan), 2);  // o_proj and down_proj
}

TEST(TPGraphRewriterAnalyze, ReportsBiasOnProjections) {
    BlockConfig config;
    config.with_bias = true;
    auto plan = GraphRewriter::analyze(make_transformer_block(config));

    size_t biased = 0;
    for (const auto& linear : plan.linears)
        biased += linear.has_bias ? 1 : 0;
    EXPECT_EQ(biased, 3u);  // q, k and v carry a bias in this configuration
}

TEST(TPGraphRewriterAnalyze, HandlesWeightsWithoutTransposeB) {
    BlockConfig config;
    config.transpose_b = false;  // weight laid out as [in, out]
    auto plan = GraphRewriter::analyze(make_transformer_block(config));

    EXPECT_EQ(plan.linears.size(), 7u);
    EXPECT_EQ(plan.num_heads, static_cast<int>(config.num_heads));
    EXPECT_EQ(plan.intermediate_size, static_cast<int>(config.intermediate));
}

TEST(TPGraphRewriterAnalyze, HandlesSingleBranchMlp) {
    BlockConfig config;
    config.gated_mlp = false;  // down(act(up(x)))
    auto plan = GraphRewriter::analyze(make_transformer_block(config));

    // q, k, v, o, up, down -- one projection fewer than the gated variant.
    EXPECT_EQ(plan.linears.size(), 6u);
    EXPECT_EQ(GraphRewriter::count_collectives(plan), 2);
}

TEST(TPGraphRewriterAnalyze, RejectsModelWithoutAttention) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 256});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256, 256}, std::vector<float>(65536, 0.1f));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(data, weights, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data});

    EXPECT_THROW(GraphRewriter::analyze(model), ov::Exception);
}

TEST(TPGraphRewriterAnalyze, RejectsProjectionWithRuntimeWeights) {
    // A weight fed by a Parameter -- a live LoRA adapter, say -- cannot be
    // sharded at compile time, and the query projection is then not found.
    BlockConfig config;
    auto model = make_transformer_block(config);

    auto weights = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::Shape{config.num_heads * config.head_dim, config.hidden});
    for (const auto& op : model->get_ordered_ops()) {
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op);
        if (matmul && matmul->get_output_partial_shape(0)[2].get_length() ==
                          static_cast<int64_t>(config.num_heads * config.head_dim)) {
            matmul->input(1).replace_source_output(weights);
            break;
        }
    }
    model->add_parameters({weights});
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(GraphRewriter::analyze(model), ov::Exception);
}

// ---------------------------------------------------------------------------
// rewrite()
// ---------------------------------------------------------------------------

class TPGraphRewriterSharding : public ::testing::TestWithParam<uint32_t> {};

TEST_P(TPGraphRewriterSharding, ProducesValidModelsForEveryRank) {
    const uint32_t world_size = GetParam();
    BlockConfig config;
    auto model = make_transformer_block(config);
    auto plan = GraphRewriter::analyze(model);

    // Shapes are re-inferred inside rewrite(); a mismatch between projections
    // sharing a dimension surfaces here.
    ASSERT_NO_THROW(rewrite_all_ranks(model, plan, world_size));
}

TEST_P(TPGraphRewriterSharding, KeepsWeightsPreSliced) {
    const uint32_t world_size = GetParam();
    BlockConfig config;
    auto model = make_transformer_block(config);
    auto plan = GraphRewriter::analyze(model);

    // A weight that cannot be pre-sliced falls back to a runtime Slice, which
    // the GPU plugin then has to constant-fold -- correct, but an order of
    // magnitude slower to compile.  The source model has no Slice at all.
    ASSERT_EQ(count_ops_of_type(model, ov::op::v8::Slice::get_type_info_static()), 0u);
    for (const auto& rank_model : rewrite_all_ranks(model, plan, world_size))
        EXPECT_EQ(count_ops_of_type(rank_model, ov::op::v8::Slice::get_type_info_static()), 0u)
            << "world_size=" << world_size;
}

TEST_P(TPGraphRewriterSharding, InsertsOneCollectivePerRowParallelProjection) {
    const uint32_t world_size = GetParam();
    BlockConfig config;
    auto model = make_transformer_block(config);
    auto plan = GraphRewriter::analyze(model);

    for (const auto& rank_model : rewrite_all_ranks(model, plan, world_size))
        EXPECT_EQ(count_ops_of_type(rank_model, ov::tp_gpu::op::TPAllReduce::get_type_info_static()), 2u);
}

INSTANTIATE_TEST_SUITE_P(TPGraphRewriter,
                         TPGraphRewriterSharding,
                         ::testing::Values(2u, 3u, 4u),
                         [](const ::testing::TestParamInfo<uint32_t>& info) {
                             return "world" + std::to_string(info.param);
                         });

// ---------------------------------------------------------------------------
// Stateful KV cache
// ---------------------------------------------------------------------------

TEST_P(TPGraphRewriterSharding, LocalizesKvCacheVariables) {
    const uint32_t world_size = GetParam();
    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);
    auto plan = GraphRewriter::analyze(model);

    for (uint32_t rank = 0; rank < world_size; ++rank) {
        auto rank_model = GraphRewriter::rewrite(model, plan, rank, world_size);

        // Whole KV heads are handed out, so the expected count is the same one
        // the rewriter derives -- spread the remainder over the first ranks.
        const auto base = config.num_kv_heads / world_size;
        const auto remainder = config.num_kv_heads % world_size;
        const int64_t expected = static_cast<int64_t>(base + (rank < remainder ? 1 : 0));

        auto variables = rank_model->get_variables();
        ASSERT_EQ(variables.size(), 2u);
        for (const auto& variable : variables) {
            const auto& shape = variable->get_info().data_shape;
            EXPECT_EQ(shape[1].get_length(), expected)
                << "variable=" << variable->get_info().variable_id << " rank=" << rank;
        }
    }
}

TEST_P(TPGraphRewriterSharding, LocalizesEveryKvCacheInitializer) {
    const uint32_t world_size = GetParam();
    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);
    auto plan = GraphRewriter::analyze(model);

    // Every ReadValue carries its own initializer chain; patching only the
    // first one leaves the rest claiming the original head count.
    auto rank_model = GraphRewriter::rewrite(model, plan, /*rank=*/0, world_size);

    const int64_t expected = static_cast<int64_t>(config.num_kv_heads / world_size +
                                                  (config.num_kv_heads % world_size ? 1 : 0));
    size_t checked = 0;
    for (const auto& op : rank_model->get_ordered_ops()) {
        auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(op);
        if (!read_value || read_value->get_input_size() == 0)
            continue;
        EXPECT_EQ(read_value->get_output_partial_shape(0)[1].get_length(), expected);
        ++checked;
    }
    EXPECT_EQ(checked, 2u) << "expected one initializer per KV cache";
}

}  // namespace ov::tp_gpu::tests
