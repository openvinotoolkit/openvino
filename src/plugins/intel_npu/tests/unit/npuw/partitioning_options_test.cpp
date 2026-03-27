// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "attention.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "partitioning/online/compiler.hpp"
#include "partitioning/partitioning.hpp"
#include "pyramid_attention.hpp"

using ov::test::npuw::ModelBuilder;
using ov::test::npuw::ModelConfig;

namespace {

::intel_npu::Config make_cfg(const ::intel_npu::Config::ConfigMap& cfg_map) {
    auto opt_desc = std::make_shared<::intel_npu::OptionsDesc>();
    ::intel_npu::registerNPUWOptions(*opt_desc);
    auto cfg = ::intel_npu::Config(opt_desc);
    cfg.update(cfg_map);
    return cfg;
}

std::filesystem::path make_unique_temp_path(const std::string& stem, const std::string& extension) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (stem + "_" + std::to_string(nonce) + extension);
}

std::shared_ptr<ov::Model> build_unary_chain_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    input->set_friendly_name("input");

    auto n1 = std::make_shared<ov::op::v1::Add>(input, input);
    n1->set_friendly_name("n1");
    auto n2 = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{1});
    n2->set_friendly_name("n2");
    auto n3 = std::make_shared<ov::op::v1::Divide>(n1, n2, true);
    n3->set_friendly_name("n3");
    auto n4 = std::make_shared<ov::op::v0::Sin>(n1);
    n4->set_friendly_name("n4");
    auto n5 = std::make_shared<ov::op::v0::Cos>(n1);
    n5->set_friendly_name("n5");
    auto n6 = std::make_shared<ov::op::v0::Sin>(n3);
    n6->set_friendly_name("n6");
    auto n7 = std::make_shared<ov::op::v0::Cos>(n3);
    n7->set_friendly_name("n7");
    auto n8 = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{n1, n4, n5, n6, n7}, -1);
    n8->set_friendly_name("n8");

    auto result = std::make_shared<ov::op::v0::Result>(n8);
    result->set_friendly_name("res");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> build_static_llm_model(const int64_t query_len, const int64_t past_len) {
    ModelConfig config;
    config.num_layers = 4;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.num_kv_heads = 4;
    config.vocab_size = 256;

    ModelBuilder mb;
    auto model = mb.build_model(config);

    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    const int64_t total = query_len + past_len;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto shape = input.get_partial_shape();
        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = {1, query_len};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = {1, total};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] =
                shape.rank().get_length() == 3 ? ov::PartialShape{3, 1, query_len} : ov::PartialShape{1, query_len};
        } else {
            shape[0] = 1;
            shape[2] = past_len;
            new_shapes[name] = shape;
        }
    }
    model->reshape(new_shapes);
    return model;
}

std::shared_ptr<ov::Model> build_static_prefill_model() {
    return build_static_llm_model(8, 8);
}

std::shared_ptr<ov::Model> build_static_generate_model() {
    return build_static_llm_model(1, 2047);
}

std::shared_ptr<ov::Model> build_repeated_model(std::size_t repetitions = 10) {
    ModelBuilder mb;
    return mb.get_model_with_repeated_blocks(repetitions);
}

// Model where a single Add output feeds two Result nodes with different names.
// Mimics the OmniThinker multi-output scenario after CutLMHead: the same tensor
// backs two named model outputs.
std::shared_ptr<ov::Model> build_multi_result_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    input->set_friendly_name("input");

    auto add = std::make_shared<ov::op::v1::Add>(input, input);
    add->set_friendly_name("add");

    auto result_a = std::make_shared<ov::op::v0::Result>(add->output(0));
    result_a->set_friendly_name("result_a");

    auto result_b = std::make_shared<ov::op::v0::Result>(add->output(0));
    result_b->set_friendly_name("result_b");

    return std::make_shared<ov::Model>(ov::ResultVector{result_a, result_b}, ov::ParameterVector{input});
}

// Same as above but with a downstream Sin reader that can be forced into a separate
// group via NPUW_ONLINE_AVOID.  This exercises the cross-group linking path when
// result_cache holds multiple entries for the same output.
std::shared_ptr<ov::Model> build_multi_result_with_downstream_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    input->set_friendly_name("input");

    auto add = std::make_shared<ov::op::v1::Add>(input, input);
    add->set_friendly_name("add");

    auto sin = std::make_shared<ov::op::v0::Sin>(add);
    sin->set_friendly_name("sin");

    // Two Results from the same Add output (different names, same data)
    auto result_a = std::make_shared<ov::op::v0::Result>(add->output(0));
    result_a->set_friendly_name("result_a");
    auto result_b = std::make_shared<ov::op::v0::Result>(add->output(0));
    result_b->set_friendly_name("result_b");

    // Third Result from Sin output
    auto result_c = std::make_shared<ov::op::v0::Result>(sin->output(0));
    result_c->set_friendly_name("result_c");

    return std::make_shared<ov::Model>(ov::ResultVector{result_a, result_b, result_c}, ov::ParameterVector{input});
}

TEST(PartitioningOptionsTest, PipelineNoneMergesUnaryModelIntoSingleGroup) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_EQ(ens.groups.size(), 1u);
}

TEST(PartitioningOptionsTest, AvoidsOnNonePipelineSplitUnaryModel) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_AVOID", "Op:Sin/NPU,Op:Cos/NPU"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_EQ(ens.groups.size(), 3u);
}

TEST(PartitioningOptionsTest, IsolateOptionTagsUnaryGroups) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "Op:Sin/compute"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_TRUE(std::any_of(ens.groups.begin(), ens.groups.end(), [](const ov::npuw::Group& group) {
        return group.gettag() == "compute";
    }));
}

TEST(PartitioningOptionsTest, DumpPlanWritesXmlFile) {
    const auto dump_path = make_unique_temp_path("npuw_partitioning_effect_dump", ".xml");

    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_DUMP_PLAN", dump_path.string()}});
    (void)ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);

    ASSERT_TRUE(std::filesystem::exists(dump_path));
    {
        std::ifstream stream(dump_path);
        std::string xml((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        EXPECT_NE(xml.find("<ensemble"), std::string::npos);
    }

    std::filesystem::remove(dump_path);
}

TEST(PartitioningOptionsTest, ComputePipelineMarksComputeGroupsAsNoFold) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "COMPUTE"}});
    auto ens = ov::npuw::online::buildPartitioning(build_static_prefill_model(), cfg);

    bool seen_compute = false;
    for (const auto& group : ens.groups) {
        if (group.gettag() == "compute") {
            seen_compute = true;
            EXPECT_TRUE(group.repeated_id.empty());
        }
    }
    EXPECT_TRUE(seen_compute);
}

TEST(PartitioningOptionsTest, SpatialPipelineDoesNotAnnotateFullPrefillModelWithoutSpatialRange) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "SPATIAL"}, {"NPUW_SPATIAL_NWAY", "16"}});
    auto partitioning = ov::npuw::getPartitioning(build_static_prefill_model(), cfg);

    bool seen_spatial = false;
    for (const auto& [_, function] : partitioning.functions) {
        seen_spatial |= function._spatial.has_value();
    }
    EXPECT_FALSE(seen_spatial);
}

TEST(PartitioningOptionsTest, DynamicAttentionRejectsUnisolatedPrefillGraph) {
    auto model = build_static_prefill_model();
    const auto attention = ov::npuw::function::Attention::from(model);

    EXPECT_FALSE(attention.has_value());
}

TEST(PartitioningOptionsTest, PyramidAttentionRejectsUnisolatedGenerateGraph) {
    auto model = build_static_generate_model();
    const auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    EXPECT_FALSE(pyramid.has_value());
}

TEST(PartitioningOptionsTest, OnlineKeepBlockSizeControlsRepeatedBlockDetection) {
    auto repeated = build_repeated_model(10);

    auto keep_cfg = make_cfg({{"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});
    auto drop_cfg = make_cfg({{"NPUW_ONLINE_KEEP_BLOCK_SIZE", "100"}});

    auto keep_ens = ov::npuw::online::buildPartitioning(repeated, keep_cfg);
    auto drop_ens = ov::npuw::online::buildPartitioning(repeated, drop_cfg);

    EXPECT_GE(keep_ens.repeated.size(), 1u);
    EXPECT_EQ(drop_ens.repeated.size(), 0u);
}

TEST(PartitioningOptionsTest, OnlineKeepBlocksControlsRepeatedBlockDetection) {
    auto repeated = build_repeated_model(3);

    auto keep_cfg = make_cfg({{"NPUW_ONLINE_KEEP_BLOCKS", "2"}, {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});
    auto drop_cfg = make_cfg({{"NPUW_ONLINE_KEEP_BLOCKS", "5"}, {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});

    auto keep_ens = ov::npuw::online::buildPartitioning(repeated, keep_cfg);
    auto drop_ens = ov::npuw::online::buildPartitioning(repeated, drop_cfg);

    EXPECT_GE(keep_ens.repeated.size(), 1u);
    EXPECT_EQ(drop_ens.repeated.size(), 0u);
}

TEST(PartitioningOptionsTest, OnlineMinSizeStopsPartitioningEarlierOnLargerGraphs) {
    auto repeated = build_repeated_model(20);

    auto compact_cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_MIN_SIZE", "10"}});
    auto early_stop_cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_MIN_SIZE", "100"}});

    auto compact_ens = ov::npuw::online::buildPartitioning(repeated, compact_cfg);
    auto early_stop_ens = ov::npuw::online::buildPartitioning(repeated, early_stop_cfg);

    EXPECT_GT(early_stop_ens.groups.size(), compact_ens.groups.size());
}

TEST(PartitioningOptionsTest, OnlineNoFoldPreventsTaggedGroupsFromBecomingRepeatedFunctions) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"},
                         {"NPUW_ONLINE_ISOLATE", "Op:Sin/compute"},
                         {"NPUW_ONLINE_NO_FOLD", "compute"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);

    EXPECT_TRUE(std::any_of(ens.groups.begin(), ens.groups.end(), [](const ov::npuw::Group& group) {
        return group.gettag() == "compute" && group.repeated_id.empty();
    }));
}

TEST(PartitioningOptionsTest, FuncallForAllPromotesUnaryGroupsToFunctions) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_FUNCALL_FOR_ALL", "YES"}});
    auto partitioning = ov::npuw::getPartitioning(build_unary_chain_model(), cfg);

    EXPECT_TRUE(std::any_of(partitioning.subgraphs.begin(), partitioning.subgraphs.end(), [](const ov::npuw::Subgraph& sg) {
        return sg._forced_to_fcall || !sg._funcall.empty() || !sg._repeated_id.empty();
    }));
}

TEST(PartitioningOptionsTest, FoldCreatesFunctionCallsForRepeatedBlocks) {
    auto cfg = make_cfg({{"NPUW_FOLD", "YES"}, {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});
    auto partitioning = ov::npuw::getPartitioning(build_repeated_model(10), cfg);

    EXPECT_FALSE(partitioning.functions.empty());
    EXPECT_TRUE(std::any_of(partitioning.subgraphs.begin(), partitioning.subgraphs.end(), [](const ov::npuw::Subgraph& sg) {
        return !sg._funcall.empty();
    }));
}

TEST(PartitioningOptionsTest, CwaiCreatesFunctionCallsForRepeatedBlocks) {
    auto cfg = make_cfg({{"NPUW_CWAI", "YES"}, {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});
    auto partitioning = ov::npuw::getPartitioning(build_repeated_model(10), cfg);

    EXPECT_FALSE(partitioning.functions.empty());
    EXPECT_TRUE(std::any_of(partitioning.subgraphs.begin(), partitioning.subgraphs.end(), [](const ov::npuw::Subgraph& sg) {
        return !sg._funcall.empty();
    }));
}

TEST(PartitioningOptionsTest, PlanFileReusesDumpedPartitioningStructure) {
    const auto plan_path = make_unique_temp_path("npuw_partitioning_effect_plan", ".xml");

    auto online_cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_DUMP_PLAN", plan_path.string()}});
    auto online_ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), online_cfg);

    auto plan_cfg = make_cfg({{"NPUW_PLAN", plan_path.string()}});
    auto partitioning = ov::npuw::getPartitioning(build_unary_chain_model(), plan_cfg);

    ASSERT_TRUE(std::filesystem::exists(plan_path));
    EXPECT_EQ(partitioning.subgraphs.size(), online_ens.groups.size());

    std::filesystem::remove(plan_path);
}

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
TEST(PartitioningOptionsTest, DumpFullWritesModelXmlIntoCurrentDirectory) {
    const auto temp_dir = make_unique_temp_path("npuw_dump_full_effect", "");
    std::filesystem::create_directories(temp_dir);
    const auto old_cwd = std::filesystem::current_path();

    auto model = build_unary_chain_model();
    model->set_friendly_name("npuw_dump_full_effect_model");

    std::filesystem::current_path(temp_dir);
    auto cfg = make_cfg({{"NPUW_DUMP_FULL", "YES"}});
    (void)ov::npuw::getPartitioning(model, cfg);
    std::filesystem::current_path(old_cwd);

    const auto dumped = temp_dir / "npuw_dump_full_effect_model.xml";
    EXPECT_TRUE(std::filesystem::exists(dumped));

    std::filesystem::remove(dumped);
    std::filesystem::remove(temp_dir / "npuw_dump_full_effect_model.bin");
    std::filesystem::remove(temp_dir);
}
#endif

bool has_result_named(const ov::npuw::Subgraph& sg, const std::string& name) {
    return std::any_of(sg._results.begin(), sg._results.end(), [&](const auto& r) {
        return r->get_friendly_name() == name;
    });
}

TEST(PartitioningOptionsTest, MultiResultFromSameOutputPreservesBothResults) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}});
    auto partitioning = ov::npuw::getPartitioning(build_multi_result_model(), cfg);

    ASSERT_EQ(partitioning.subgraphs.size(), 1u);
    ASSERT_EQ(partitioning.subgraphs[0]._results.size(), 2u);
    EXPECT_TRUE(has_result_named(partitioning.subgraphs[0], "result_a"));
    EXPECT_TRUE(has_result_named(partitioning.subgraphs[0], "result_b"));
}

TEST(PartitioningOptionsTest, MultiResultFromSameOutputConnectsDownstreamAcrossGroups) {
    // Sin is avoided to force a second group; Add's output feeds two Results AND the Sin reader.
    // Tests that result_cache (now a vector) correctly handles linking when multiple Results
    // exist from the same output.
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_AVOID", "Op:Sin/NPU"}});
    auto partitioning = ov::npuw::getPartitioning(build_multi_result_with_downstream_model(), cfg);

    // Expect at least 2 subgraphs (Add group, Sin group)
    ASSERT_GE(partitioning.subgraphs.size(), 2u);

    // Find the subgraph containing the two Results from Add's output
    const ov::npuw::Subgraph* add_sg = nullptr;
    for (const auto& sg : partitioning.subgraphs) {
        if (has_result_named(sg, "result_a")) {
            add_sg = &sg;
            break;
        }
    }
    ASSERT_NE(add_sg, nullptr) << "No subgraph contains result_a";
    EXPECT_TRUE(has_result_named(*add_sg, "result_b"));

    // result_c (from Sin) must exist and be in a *different* subgraph
    const ov::npuw::Subgraph* sin_sg = nullptr;
    for (const auto& sg : partitioning.subgraphs) {
        if (has_result_named(sg, "result_c")) {
            sin_sg = &sg;
            break;
        }
    }
    ASSERT_NE(sin_sg, nullptr) << "No subgraph contains result_c";
    EXPECT_NE(sin_sg, add_sg) << "result_c must be in a different subgraph than result_a/result_b";

    // A cross-group link must exist (Sin group reads Add's output)
    EXPECT_FALSE(partitioning.input_to_prev_output.empty());
}

TEST(PartitioningOptionsTest, RepPipelinePreservesMultiResultInLLMModel) {
    // Build a static LLM model (4 repeated transformer layers) and add a second
    // Result from the same output as the existing logits Result. This mimics the
    // OmniThinker/CutLMHead scenario on a model with repeated blocks detected by
    // the REP pipeline.
    auto model = build_static_prefill_model();

    // Find the "logits" Result and duplicate it: attach a second Result to the same source.
    // This mimics the CutLMHead scenario where the output before the LM head already had
    // a Result, and cutting the head creates another Result from the same tensor.
    std::shared_ptr<ov::op::v0::Result> logits_result;
    for (const auto& r : model->get_results()) {
        if (r->get_friendly_name() == "logits") {
            logits_result = r;
            break;
        }
    }
    ASSERT_NE(logits_result, nullptr) << "Model must have a 'logits' Result";
    auto source_output = logits_result->input(0).get_source_output();
    auto extra_result = std::make_shared<ov::op::v0::Result>(source_output);
    extra_result->set_friendly_name("extra_output");
    model->add_results({extra_result});

    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"}});
    auto partitioning = ov::npuw::getPartitioning(model, cfg);

    // The tail subgraph must contain both the original Result and the extra_output
    // Result, verified by their friendly names.
    const ov::npuw::Subgraph* tail_sg = nullptr;
    for (const auto& sg : partitioning.subgraphs) {
        if (has_result_named(sg, "extra_output")) {
            tail_sg = &sg;
            break;
        }
    }
    ASSERT_NE(tail_sg, nullptr) << "No subgraph contains extra_output";
    EXPECT_TRUE(has_result_named(*tail_sg, "logits"));
}

}  // namespace
