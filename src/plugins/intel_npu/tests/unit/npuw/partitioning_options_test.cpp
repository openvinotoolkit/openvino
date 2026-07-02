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
using ov::test::npuw::LLMConfig;

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
    LLMConfig config;
    config.num_layers = 4;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.num_kv_heads = 4;
    config.vocab_size = 256;

    ModelBuilder mb;
    auto model = mb.build_llm(config);

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

std::shared_ptr<ov::Model> build_static_attention_llm_model() {
    LLMConfig config;
    config.num_layers = 2;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.num_kv_heads = 2;
    config.vocab_size = 256;
    config.force_gqa_broadcast = true;

    ModelBuilder mb;
    auto model = mb.build_llm(config);

    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    constexpr int64_t query_len = 4;
    constexpr int64_t past_len = 8;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto shape = input.get_partial_shape();
        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = {1, query_len};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = {1, query_len + past_len};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] = {1, query_len};
        } else {
            shape[0] = 1;
            shape[2] = past_len;
            new_shapes[name] = shape;
        }
    }
    model->reshape(new_shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> build_static_attention_mixed_llm_model() {
    LLMConfig config;
    config.num_layers = 4;
    config.hidden_size = 64;
    config.num_heads = 4;
    config.head_dim = 16;
    config.num_kv_heads = 2;
    config.vocab_size = 256;
    config.force_gqa_broadcast = true;

    ModelBuilder mb;
    auto model = mb.build_llm(config);

    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    constexpr int64_t query_len = 4;
    constexpr int64_t past_len = 8;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        auto shape = input.get_partial_shape();
        if (name.find("input_ids") != std::string::npos || name.find("token_type_ids") != std::string::npos) {
            new_shapes[name] = {1, query_len};
        } else if (name.find("attention_mask") != std::string::npos) {
            new_shapes[name] = {1, query_len + past_len};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shapes[name] = {1, query_len};
        } else {
            shape[0] = 1;
            shape[2] = past_len;
            new_shapes[name] = shape;
        }
    }
    model->reshape(new_shapes);
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> build_repeated_model(std::size_t repetitions = 10) {
    ModelBuilder mb;
    return mb.get_model_with_repeated_blocks(repetitions);
}

// Build a model with N repetitions of (Relu -> Sigmoid -> Tanh).
// Each op type forms its own isolated tag so mergeTriangles cannot merge the
// three families into one combined repeating block.
std::shared_ptr<ov::Model> build_abc_attn_model(std::size_t repetitions = 30) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32});
    input->set_friendly_name("input");

    std::shared_ptr<ov::Node> prev = input;
    for (std::size_t i = 0; i < repetitions; ++i) {
        auto relu = std::make_shared<ov::op::v0::Relu>(prev);
        relu->set_friendly_name("relu_" + std::to_string(i));
        auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(relu);
        sigmoid->set_friendly_name("sigmoid_" + std::to_string(i));
        auto tanh = std::make_shared<ov::op::v0::Tanh>(sigmoid);
        tanh->set_friendly_name("tanh_" + std::to_string(i));
        prev = tanh;
    }

    auto result = std::make_shared<ov::op::v0::Result>(prev);
    result->set_friendly_name("output");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
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

TEST(PartitioningOptionsTest, FoldOnlyProcessesTaggedRepeatedFamiliesWithoutCwai) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"},
                         {"NPUW_ONLINE_ISOLATE", "ATTN"},
                         {"NPUW_ATTN", "DYNAMIC"},
                         {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                         {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
                         {"NPUW_FOLD_ONLY", "attn"}});
    auto partitioning = ov::npuw::getPartitioning(build_static_attention_llm_model(), cfg);

    EXPECT_FALSE(partitioning.functions.empty());
    EXPECT_TRUE(std::any_of(partitioning.subgraphs.begin(), partitioning.subgraphs.end(), [](const ov::npuw::Subgraph& sg) {
        return !sg._funcall.empty();
    }));
}

TEST(PartitioningOptionsTest, FoldOnlyAndCwaiProcessTaggedAndUntaggedRepeatedFamilies) {
    const auto base_cfg = ::intel_npu::Config::ConfigMap{{"NPUW_ONLINE_PIPELINE", "REP"},
                                                         {"NPUW_ONLINE_ISOLATE", "COMPUTE,ATTN"},
                                                         {"NPUW_ONLINE_KEEP_BLOCKS", "2"},
                                                         {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
                                                         {"NPUW_FOLD_ONLY", "attn"},
                                                         {"NPUW_ATTN", "STATIC"}};
    auto fold_only_cfg = make_cfg(base_cfg);
    auto fold_only_partitioning = ov::npuw::getPartitioning(build_static_attention_mixed_llm_model(), fold_only_cfg);

    auto mixed_cfg = base_cfg;
    mixed_cfg["NPUW_CWAI"] = "YES";
    auto mixed_cfg_obj = make_cfg(mixed_cfg);
    auto mixed_partitioning = ov::npuw::getPartitioning(build_static_attention_mixed_llm_model(), mixed_cfg_obj);

    EXPECT_TRUE(std::any_of(mixed_partitioning.functions.begin(), mixed_partitioning.functions.end(), [](const auto& func) {
        return func.second.gettag() == "attn";
    }));

    const auto has_cwai_function = [](const ov::npuw::Partitioning& partitioning) {
        return std::any_of(partitioning.functions.begin(), partitioning.functions.end(), [](const auto& func) {
            return func.first.find("__") != std::string::npos;
        });
    };
    const auto has_cwai_funcall = [](const ov::npuw::Partitioning& partitioning) {
        return std::any_of(partitioning.subgraphs.begin(), partitioning.subgraphs.end(), [](const ov::npuw::Subgraph& sg) {
            return sg._funcall.find("__") != std::string::npos;
        });
    };

    EXPECT_FALSE(has_cwai_function(fold_only_partitioning));
    EXPECT_FALSE(has_cwai_funcall(fold_only_partitioning));
    EXPECT_TRUE(has_cwai_function(mixed_partitioning));
    EXPECT_TRUE(has_cwai_funcall(mixed_partitioning));
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

// Isolate three op families with distinct tags so that mergeTriangles cannot
// collapse them into a single combined repeating block:
//   blockA = Relu, blockB = Sigmoid, attn = Tanh
// With N=30 we get 3*N=90 frozen groups after repeatedBlocks.
static const ::intel_npu::Config::ConfigMap abc_attn_base_cfg = {
    {"NPUW_ONLINE_PIPELINE", "REP"},
    {"NPUW_ONLINE_ISOLATE", "Op:Relu/blockA,Op:Sigmoid/blockB,Op:Tanh/attn"},
    {"NPUW_ONLINE_KEEP_BLOCKS", "3"},
    {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
    {"NPUW_FOLD_ONLY", "attn"},
};

TEST(PartitioningOptionsTest, FoldOnlyWithIsolatedTagsProducesExpectedSubgraphCount) {
    // Baseline: FOLD_ONLY folds the 30 attn blocks; blockA and blockB remain
    // as individual non-folded subgraphs → 3*30 = 90 subgraphs, 30 with funcalls.
    constexpr std::size_t N = 30;
    auto cfg = make_cfg(abc_attn_base_cfg);
    auto partitioning = ov::npuw::getPartitioning(build_abc_attn_model(N), cfg);

    EXPECT_EQ(partitioning.subgraphs.size(), 3u * N);

    std::size_t folded = std::count_if(partitioning.subgraphs.begin(),
                                       partitioning.subgraphs.end(),
                                       [](const ov::npuw::Subgraph& sg) {
                                           return !sg._funcall.empty();
                                       });
    EXPECT_EQ(folded, N);
}

TEST(PartitioningOptionsTest, FuseUnfoldedMergesNonFoldOnlyRepeatedBlocks) {
    // With NPUW_FUSE_UNFOLDED, blockA (Relu) and blockB (Sigmoid) groups lose
    // their reptag and are merged by fuseRemnants (frozen attn blocks act as
    // barriers).  Result: 30 merged(blockA+blockB) + 30 folded attn = 2*30 = 60.
    constexpr std::size_t N = 30;
    auto ext_cfg = abc_attn_base_cfg;
    ext_cfg["NPUW_FUSE_UNFOLDED"] = "YES";
    auto cfg = make_cfg(ext_cfg);
    auto partitioning = ov::npuw::getPartitioning(build_abc_attn_model(N), cfg);

    EXPECT_EQ(partitioning.subgraphs.size(), 2u * N);

    std::size_t folded = std::count_if(partitioning.subgraphs.begin(),
                                       partitioning.subgraphs.end(),
                                       [](const ov::npuw::Subgraph& sg) {
                                           return !sg._funcall.empty();
                                       });
    EXPECT_EQ(folded, N);
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

}  // namespace
