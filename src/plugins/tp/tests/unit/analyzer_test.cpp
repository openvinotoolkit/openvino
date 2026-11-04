// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The L1 analysis against the rewriter it replaces: on every synthetic block, for 2, 3 and 4
// ranks, the block passes claim the same projections with the same roles and collectives, the
// same vocabulary projection, the same KV-cache variables and the same geometry.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <map>
#include <set>
#include <string>

#include "common_test_utils/test_assertions.hpp"
#include "graph_rewriter.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/runtime/core.hpp"
#include "tp_test_models.hpp"
#include "tp_transformations/analyzer.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

struct Variant {
    std::string name;
    BlockConfig config;
    bool paged = false;
    size_t vocab = 0;
};

std::shared_ptr<ov::Model> build(const Variant& variant) {
    if (variant.paged)
        return make_paged_attention_block(variant.config);
    if (variant.vocab != 0)
        return make_block_with_lm_head(variant.config, variant.vocab);
    return make_transformer_block(variant.config);
}

std::vector<Variant> variants() {
    std::vector<Variant> result;
    result.push_back({"default", BlockConfig{}});
    BlockConfig c;
    c.with_bias = true;
    result.push_back({"bias", c});
    c = BlockConfig{};
    c.transpose_b = false;
    result.push_back({"in_out_layout", c});
    c = BlockConfig{};
    c.gated_mlp = false;
    result.push_back({"dense_mlp", c});
    c = BlockConfig{};
    c.stateful = true;
    result.push_back({"stateful", c});
    c = BlockConfig{};
    c.group_size = -1;
    result.push_back({"per_channel", c});
    c = BlockConfig{};
    c.weights_precision = ov::element::u8;
    result.push_back({"u8", c});
    c = BlockConfig{};
    c.weights_precision = ov::element::f32;
    result.push_back({"f32", c});
    result.push_back({"lm_head", BlockConfig{}, false, 1024});
    result.push_back({"lm_head_uneven", BlockConfig{}, false, 1022});
    result.push_back({"paged", BlockConfig{}, true});
    return result;
}

/// The analysis of `original` for `world_size` ranks against the rewriter's.
void expect_rewriter_claims(const std::shared_ptr<ov::Model>& original, size_t world_size) {
    const auto plan = GraphRewriter::analyze(original);

    auto analyzed = original->clone();
    auto report = std::make_shared<TPAnalysisReport>();
    TPLayerAnalyzer(TPAnalyzerConfig{world_size, true}, report).run_on_model(analyzed);

    std::map<std::string, ProjectionRole> roles;
    for (const auto& node : analyzed->get_ordered_ops()) {
        if (const auto* role = get_role(*node))
            roles[node->get_friendly_name()] = *role;
    }

    for (const auto& linear : plan.linears) {
        ASSERT_EQ(roles.count(linear.matmul_name), 1u) << linear.matmul_name << " is not claimed";
        const auto& role = roles.at(linear.matmul_name);
        EXPECT_EQ(role.strategy,
                  linear.is_column_parallel ? ShardStrategy::COLUMN_PARALLEL : ShardStrategy::ROW_PARALLEL)
            << linear.matmul_name;
        EXPECT_EQ(role.collective, linear.is_column_parallel ? Collective::NONE : Collective::ALL_REDUCE)
            << linear.matmul_name;
        roles.erase(linear.matmul_name);
    }

    const bool lm_head = GraphRewriter::shards_lm_head(plan, static_cast<int>(world_size));
    EXPECT_EQ(report->lm_head, lm_head);
    if (lm_head) {
        ASSERT_EQ(roles.count(plan.lm_head_name), 1u);
        EXPECT_EQ(roles.at(plan.lm_head_name).collective, Collective::GATHER);
        roles.erase(plan.lm_head_name);
    }
    EXPECT_TRUE(roles.empty()) << roles.size() << " projections claimed beyond the rewriter's, e.g. "
                               << roles.begin()->first;

    EXPECT_EQ(report->attention_blocks, static_cast<size_t>(plan.num_layers));
    EXPECT_EQ(report->num_heads, plan.num_heads);
    EXPECT_EQ(report->num_kv_heads, plan.num_kv_heads);
    EXPECT_EQ(report->head_dim, plan.head_dim);
    EXPECT_EQ(report->intermediate_size, plan.intermediate_size);

    // Weights the rewriter cuts with a Slice at run time, on any rank.
    std::set<std::string> expected_slices;
    for (size_t rank = 0; rank < world_size; ++rank) {
        const auto rank_model =
            GraphRewriter::rewrite(original, plan, static_cast<uint32_t>(rank), static_cast<uint32_t>(world_size));
        for (const auto& node : rank_model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v0::MatMul>(node) && ov::is_type<ov::op::v8::Slice>(node->get_input_node_ptr(1)))
                expected_slices.insert(node->get_friendly_name());
        }
    }
    const std::set<std::string> slices(report->runtime_slices.begin(), report->runtime_slices.end());
    EXPECT_EQ(slices, expected_slices);

    // The only linear layer left whole: a vocabulary projection that does not divide.
    const bool whole_lm_head = !plan.lm_head_name.empty() && !lm_head;
    EXPECT_EQ(report->unclaimed, whole_lm_head ? std::vector<std::string>{plan.lm_head_name} : std::vector<std::string>{});

    auto expected_variables = GraphRewriter::sharded_state_ids(original, plan);
    auto variables = report->sharded_variables;
    std::sort(expected_variables.begin(), expected_variables.end());
    std::sort(variables.begin(), variables.end());
    EXPECT_EQ(variables, expected_variables);
}

class TPAnalyzerGoldenTest : public ::testing::TestWithParam<std::tuple<Variant, size_t>> {};

TEST_P(TPAnalyzerGoldenTest, ClaimsWhatTheRewriterClaims) {
    const auto& [variant, world_size] = GetParam();
    expect_rewriter_claims(build(variant), world_size);
}

// Real models, listed in OV_TP_GOLDEN_MODELS (IR paths separated by ':').
TEST(TPAnalyzerGoldenModels, ClaimWhatTheRewriterClaims) {
    const char* list = std::getenv("OV_TP_GOLDEN_MODELS");
    if (list == nullptr || *list == 0)
        GTEST_SKIP() << "set OV_TP_GOLDEN_MODELS to IR paths separated by ':'";
    ov::Core core;
    std::stringstream paths(list);
    std::string path;
    while (std::getline(paths, path, ':')) {
        const auto model = core.read_model(path);
        for (const size_t world_size : {2u, 4u}) {
            SCOPED_TRACE(path + " on " + std::to_string(world_size) + " ranks");
            expect_rewriter_claims(model, world_size);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TPAnalyzer,
                         TPAnalyzerGoldenTest,
                         ::testing::Combine(::testing::ValuesIn(variants()), ::testing::Values(2u, 3u, 4u)),
                         [](const ::testing::TestParamInfo<std::tuple<Variant, size_t>>& info) {
                             return std::get<0>(info.param).name + "_world" + std::to_string(std::get<1>(info.param));
                         });

TEST(TPAnalyzerTest, RejectsAModelWithoutAttention) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 256});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256, 256}, std::vector<float>(65536, 0.1f));
    auto matmul = std::make_shared<ov::op::v0::MatMul>(data, weights, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{data});
    OV_EXPECT_THROW(TPLayerAnalyzer(TPAnalyzerConfig{2, true}).run_on_model(model),
                    ov::Exception,
                    HasSubstr("Could not identify transformer layers"));
}

// A shared weight (a tied embedding and LM head) is cut for the projection only.
TEST(TPAnalyzerTest, CutsASharedWeightForTheProjectionOnly) {
    auto model = make_block_with_lm_head(BlockConfig{}, 1024);
    std::shared_ptr<ov::Node> lm_head;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "lm_head")
            lm_head = node;
    }
    ASSERT_NE(lm_head, nullptr);
    const auto weight = lm_head->input_value(1);
    auto other = std::make_shared<ov::op::v0::Result>(weight);
    model->add_results({other});

    TPLayerAnalyzer(TPAnalyzerConfig{2, true}).run_on_model(model);
    EXPECT_EQ(get_shard(*other->get_input_node_ptr(0)), nullptr) << "the other reader's weight was claimed";
    EXPECT_NE(lm_head->get_input_node_ptr(1), other->get_input_node_ptr(0));
}

}  // namespace
}  // namespace ov::tp::tests
