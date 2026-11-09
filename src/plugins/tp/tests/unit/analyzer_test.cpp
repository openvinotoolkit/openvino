// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "l1_test_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/tp_collective.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::HasSubstr;

/// Row-parallel projections among `claimed`; checks every role's collective on the way.
size_t row_parallel(const std::map<std::string, ProjectionRole>& claimed) {
    size_t count = 0;
    for (const auto& [name, role] : claimed) {
        if (role.collective == Collective::GATHER)
            continue;
        const bool row = role.strategy == ShardStrategy::ROW_PARALLEL;
        EXPECT_EQ(role.collective, row ? Collective::ALL_REDUCE : Collective::NONE) << name;
        count += row ? 1 : 0;
    }
    return count;
}

}  // namespace

TEST(TPAnalyzerTest, FindsTheProjectionsAndTheGeometry) {
    BlockConfig config;
    auto model = make_transformer_block(config);
    const auto report = analyze(model, 2);

    EXPECT_EQ(report->attention_blocks, 1u);
    EXPECT_EQ(report->mlp_blocks, 1u);
    EXPECT_EQ(report->num_heads, static_cast<int64_t>(config.num_heads));
    EXPECT_EQ(report->num_kv_heads, static_cast<int64_t>(config.num_kv_heads));
    EXPECT_EQ(report->head_dim, static_cast<int64_t>(config.head_dim));
    EXPECT_EQ(report->intermediate_size, static_cast<int64_t>(config.intermediate));
    EXPECT_FALSE(report->lm_head);
    EXPECT_TRUE(report->unclaimed.empty());
    EXPECT_TRUE(report->runtime_slices.empty());
    EXPECT_TRUE(report->sharded_variables.empty());

    // q, k, v, gate, up column-parallel; o and down row-parallel.
    const auto claimed = roles(model);
    EXPECT_EQ(claimed.size(), 7u);
    EXPECT_EQ(row_parallel(claimed), 2u);
}

TEST(TPAnalyzerTest, SplitsTheBiasOfColumnParallelProjections) {
    BlockConfig config;
    config.with_bias = true;
    auto model = make_transformer_block(config);
    analyze(model, 2);

    size_t split = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::Add>(node))
            continue;
        for (const auto& input : node->input_values())
            split += ov::is_type<ov::op::v0::Constant>(input.get_node()) && get_shard(*input.get_node()) ? 1 : 0;
    }
    EXPECT_EQ(split, 3u);  // q, k and v carry a bias in this configuration
}

TEST(TPAnalyzerTest, HandlesWeightsWithoutTransposeB) {
    BlockConfig config;
    config.transpose_b = false;  // weight laid out as [in, out]
    auto model = make_transformer_block(config);
    const auto report = analyze(model, 2);

    EXPECT_EQ(roles(model).size(), 7u);
    EXPECT_EQ(report->num_heads, static_cast<int64_t>(config.num_heads));
    EXPECT_EQ(report->intermediate_size, static_cast<int64_t>(config.intermediate));
}

TEST(TPAnalyzerTest, HandlesADenseMlp) {
    BlockConfig config;
    config.gated_mlp = false;  // down(act(up(x)))
    auto model = make_transformer_block(config);
    analyze(model, 2);

    const auto claimed = roles(model);
    EXPECT_EQ(claimed.size(), 6u);
    EXPECT_EQ(row_parallel(claimed), 2u);
}

TEST(TPAnalyzerTest, RecognizesPagedAttention) {
    BlockConfig config;
    auto model = make_paged_attention_block(config);
    const auto report = analyze(model, 2);

    EXPECT_EQ(report->attention_blocks, 1u);
    EXPECT_EQ(report->num_heads, static_cast<int64_t>(config.num_heads));
    EXPECT_EQ(report->num_kv_heads, static_cast<int64_t>(config.num_kv_heads));
    EXPECT_EQ(report->head_dim, static_cast<int64_t>(config.head_dim));
    EXPECT_EQ(roles(model).size(), 7u);
    // The cache took the role of the ReadValue / Assign pairs: no state to split.
    EXPECT_TRUE(report->sharded_variables.empty());
}

TEST(TPAnalyzerTest, ReportsTheKvCacheVariables) {
    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);
    auto variables = analyze(model, 2)->sharded_variables;
    std::sort(variables.begin(), variables.end());
    EXPECT_EQ(variables, (std::vector<std::string>{"past_key_values.0.key", "past_key_values.0.value"}));
}

TEST(TPAnalyzerTest, RejectsAProjectionWithRuntimeWeights) {
    // A weight fed by a Parameter -- a live LoRA adapter, say -- cannot be split at compile time.
    BlockConfig config;
    auto model = make_transformer_block(config);
    auto weights =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{config.num_heads * config.head_dim, config.hidden});
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

    EXPECT_THROW(analyze(model, 2), ov::Exception);
}

TEST(TPAnalyzerTest, RejectsMoreRanksThanKvHeads) {
    BlockConfig config;  // 4 KV heads
    auto model = make_transformer_block(config);
    EXPECT_THROW(analyze(model, config.num_kv_heads + 1), ov::Exception);
}

// Two quantization groups of the MLP cannot give three ranks one each.
TEST(TPAnalyzerTest, RejectsMoreRanksThanQuantizationGroups) {
    BlockConfig config;
    config.group_size = 128;
    config.intermediate = 256;
    EXPECT_NO_THROW(analyze(make_transformer_block(config), 2));
    OV_EXPECT_THROW(analyze(make_transformer_block(config), 3),
                    ov::Exception,
                    ::testing::HasSubstr("2 unit(s) for 3 ranks: a rank would get none. Use at most 2 devices"));
}

// Weights of whole bytes and 4-bit ones are split; other packings are rejected, naming the weight.
TEST(TPAnalyzerTest, SplitsOnlyWeightsOfWholeBytesOrFourBits) {
    auto with_weight_type = [](ov::element::Type type) {
        auto model = make_transformer_block(BlockConfig{});
        for (const auto& op : model->get_ordered_ops()) {
            const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (constant && constant->get_element_type() == ov::element::u4 && constant->get_shape().size() == 3) {
                const auto replacement = ov::op::v0::Constant::create(type, constant->get_shape(), {0});
                replacement->set_friendly_name("packed_weight");
                constant->output(0).replace(replacement->output(0));
                break;
            }
        }
        model->validate_nodes_and_infer_types();
        return model;
    };
    for (const auto type : {ov::element::u8, ov::element::i8, ov::element::f16, ov::element::i4, ov::element::nf4})
        EXPECT_NO_THROW(analyze(with_weight_type(type), 2)) << type;
    for (const auto type : {ov::element::u1, ov::element::u2, ov::element::u3, ov::element::u6}) {
        OV_EXPECT_THROW(analyze(with_weight_type(type), 2),
                        ov::Exception,
                        ::testing::HasSubstr("weight constant 'packed_weight' of element type " +
                                             std::string(type.get_type_name())));
    }
}

TEST(TPAnalyzerTest, SplitsTheVocabularyProjection) {
    auto model = make_block_with_lm_head(BlockConfig{}, 1024);
    const auto report = analyze(model, 4);

    EXPECT_TRUE(report->lm_head);
    EXPECT_TRUE(report->unclaimed.empty());
    const auto claimed = roles(model);
    ASSERT_EQ(claimed.count("lm_head"), 1u);
    EXPECT_EQ(claimed.at("lm_head").strategy, ShardStrategy::COLUMN_PARALLEL);
    EXPECT_EQ(claimed.at("lm_head").collective, Collective::GATHER);
}

TEST(TPAnalyzerTest, LeavesAVocabularyThatDoesNotDivideWhole) {
    // 1022 = 2 * 7 * 73: splits in two, never in four.
    auto two = make_block_with_lm_head(BlockConfig{}, 1022);
    EXPECT_TRUE(analyze(two, 2)->lm_head);

    auto four = make_block_with_lm_head(BlockConfig{}, 1022);
    const auto report = analyze(four, 4);
    EXPECT_FALSE(report->lm_head);
    EXPECT_EQ(report->unclaimed, std::vector<std::string>{"lm_head"});
    EXPECT_EQ(roles(four).count("lm_head"), 0u);
}

TEST(TPAnalyzerTest, LeavesTheVocabularyProjectionWholeOnRequest) {
    auto model = make_block_with_lm_head(BlockConfig{}, 1024);
    const auto report = analyze(model, 2, /*shard_lm_head=*/false);
    EXPECT_FALSE(report->lm_head);
    EXPECT_EQ(report->unclaimed, std::vector<std::string>{"lm_head"});
}

// Real models, listed in OV_TP_TEST_MODELS (IR paths separated by ':').
TEST(TPAnalyzerModels, BuildEveryRankModel) {
    const char* list = std::getenv("OV_TP_TEST_MODELS");
    if (list == nullptr || *list == 0)
        GTEST_SKIP() << "set OV_TP_TEST_MODELS to IR paths separated by ':'";
    ov::Core core;
    std::stringstream paths(list);
    std::string path;
    while (std::getline(paths, path, ':')) {
        const auto model = core.read_model(path);
        const auto tables = gather_tables(model);
        for (const size_t world_size : {2u, 4u}) {
            SCOPED_TRACE(path + " on " + std::to_string(world_size) + " ranks");
            auto analyzed = model->clone();
            const auto report = analyze(analyzed, world_size);
            EXPECT_EQ(report->attention_blocks, report->mlp_blocks);
            EXPECT_TRUE(report->unclaimed.empty()) << report->unclaimed.front();
            for (size_t rank = 0; rank < world_size; ++rank) {
                const auto model_of_rank = rank_model(analyzed, rank, world_size);
                EXPECT_EQ(count_ops_of_type(model_of_rank, ov::op::internal::TPAllReduce::get_type_info_static()),
                          2 * report->attention_blocks);
                EXPECT_EQ(count_ops_of_type(model_of_rank, ov::op::internal::TPGather::get_type_info_static()),
                          report->lm_head ? 1u : 0u);
                EXPECT_EQ(gather_tables(model_of_rank), tables) << "an embedding table was cut";
            }
        }
    }
}

TEST(TPAnalyzerTest, GivesTheRanksDifferentSlices) {
    auto analyzed = make_transformer_block(BlockConfig{});
    analyze(analyzed, 2);
    const auto comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES).enable(FunctionsComparator::ATTRIBUTES);
    EXPECT_FALSE(comparator.compare(rank_model(analyzed, 0, 2), rank_model(analyzed, 1, 2)).valid);
}

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

// A tied embedding: the table the Gather reads stays whole on every rank, the LM head is cut.
TEST(TPAnalyzerTest, KeepsATiedEmbeddingTableWhole) {
    constexpr size_t vocab = 1024;
    auto model = make_block_with_lm_head(BlockConfig{}, vocab);
    std::shared_ptr<ov::Node> lm_head;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "lm_head")
            lm_head = node;
    }
    ASSERT_NE(lm_head, nullptr);
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto embedding = std::make_shared<ov::op::v8::Gather>(lm_head->input_value(1), ids, axis);
    embedding->set_friendly_name("embedding");
    model->add_parameters({ids});
    model->add_results({std::make_shared<ov::op::v0::Result>(embedding)});

    const auto tables = gather_tables(model);
    ASSERT_EQ(tables.at("embedding"), (ov::PartialShape{vocab, BlockConfig{}.hidden}));
    auto report = std::make_shared<TPAnalysisReport>();
    TPLayerAnalyzer(TPAnalyzerConfig{2, true}, report).run_on_model(model);
    ASSERT_TRUE(report->lm_head);
    for (size_t rank = 0; rank < 2; ++rank) {
        const auto model_of_rank = rank_model(model, rank, 2);
        EXPECT_EQ(gather_tables(model_of_rank), tables) << "rank " << rank;
        for (const auto& node : model_of_rank->get_ordered_ops()) {
            if (node->get_friendly_name() == "lm_head") {
                EXPECT_EQ(node->get_input_partial_shape(1), (ov::PartialShape{vocab / 2, BlockConfig{}.hidden}));
            }
        }
    }
}

// The report prices the cut copy a tied LM head adds on every rank; an LM head with a weight of
// its own adds none.
TEST(TPAnalyzerTest, ReportsTheCopyATiedLmHeadAdds) {
    constexpr size_t vocab = 1024;
    const auto tied = [] {
        auto model = make_block_with_lm_head(BlockConfig{}, vocab);
        for (const auto& node : model->get_ordered_ops()) {
            if (node->get_friendly_name() != "lm_head")
                continue;
            auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
            auto embedding = std::make_shared<ov::op::v8::Gather>(node->input_value(1), ids, axis);
            model->add_parameters({ids});
            model->add_results({std::make_shared<ov::op::v0::Result>(embedding)});
            break;
        }
        return model;
    };
    EXPECT_EQ(analyze(make_block_with_lm_head(BlockConfig{}, vocab), 2)->lm_head_copy_bytes, 0u);
    const auto on_two = analyze(tied(), 2)->lm_head_copy_bytes;
    // At least half of the u4 table itself.
    EXPECT_GE(on_two, vocab * BlockConfig{}.hidden / 2 / 2);
    EXPECT_EQ(analyze(tied(), 4)->lm_head_copy_bytes, on_two / 2);
}

}  // namespace ov::tp::tests
