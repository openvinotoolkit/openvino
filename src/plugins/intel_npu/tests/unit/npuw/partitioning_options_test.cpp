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
#include <set>
#include <string>
#include <tuple>

#include "attention.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "model_builder.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
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

// Build a chain with exactly two Relu and two Sigmoid ops, but only ONE
// Relu -> Sigmoid adjacency. The unary filler ops are all distinct types so they
// never form repeating families of their own, and the chain is long enough to stay
// above NPUW_ONLINE_MIN_SIZE (which is clamped to 10 groups).
std::shared_ptr<ov::Model> build_single_pair_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32});
    input->set_friendly_name("input");

    std::shared_ptr<ov::Node> prev = input;
    const auto chain = [&](const std::shared_ptr<ov::Node>& node, const std::string& name) {
        node->set_friendly_name(name);
        prev = node;
    };

    chain(std::make_shared<ov::op::v0::Relu>(prev), "relu_0");
    chain(std::make_shared<ov::op::v0::Sigmoid>(prev), "sigmoid_0");
    chain(std::make_shared<ov::op::v0::Abs>(prev), "abs_0");
    chain(std::make_shared<ov::op::v0::Ceiling>(prev), "ceil_0");
    chain(std::make_shared<ov::op::v0::Floor>(prev), "floor_0");
    chain(std::make_shared<ov::op::v0::Sqrt>(prev), "sqrt_0");
    chain(std::make_shared<ov::op::v0::Exp>(prev), "exp_0");
    chain(std::make_shared<ov::op::v0::Log>(prev), "log_0");
    chain(std::make_shared<ov::op::v0::Sin>(prev), "sin_0");
    chain(std::make_shared<ov::op::v0::Cos>(prev), "cos_0");
    chain(std::make_shared<ov::op::v0::Negative>(prev), "neg_0");
    chain(std::make_shared<ov::op::v0::Relu>(prev), "relu_1");
    chain(std::make_shared<ov::op::v0::Tanh>(prev), "tanh_0");
    chain(std::make_shared<ov::op::v0::Sigmoid>(prev), "sigmoid_1");

    auto result = std::make_shared<ov::op::v0::Result>(prev);
    result->set_friendly_name("output");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

// Partition boundaries as an order-independent set of layer sets, so the assertions
// don't depend on the (unspecified) group or intra-group layer ordering.
std::set<std::set<std::string>> partition_boundaries(const ov::npuw::Ensemble& ens) {
    std::set<std::set<std::string>> boundaries;
    for (const auto& group : ens.groups) {
        boundaries.emplace(group.all_layers.begin(), group.all_layers.end());
    }
    return boundaries;
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
struct ShapeConsumerModel {
    std::shared_ptr<ov::Model> model;
    std::vector<bool> eliminated;
};

ShapeConsumerModel build_shape_consumer_model(unsigned eliminated_mask) {
    ShapeConsumerModel built;
    ov::ParameterVector inputs;
    ov::ResultVector results;
    for (unsigned i = 0; i < 3; ++i) {
        const auto suffix = std::to_string(i);
        const bool eliminated = (eliminated_mask & (1u << i)) != 0;
        auto input = std::make_shared<ov::op::v0::Parameter>(eliminated ? ov::element::f32 : ov::element::i64,
                                                             eliminated ? ov::Shape{2, 3} : ov::Shape{2});
        input->set_friendly_name("input_" + suffix);
        std::shared_ptr<ov::Node> shape = input;
        if (eliminated) {
            shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
        }
        auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto gather = std::make_shared<ov::op::v8::Gather>(shape, indices, axis);
        gather->set_friendly_name("gather_" + suffix);
        if (eliminated) {
            ov::util::evaluate_both_bounds(gather->output(0));
        }
        auto value = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto consumer = std::make_shared<ov::op::v1::Add>(gather, value);
        consumer->set_friendly_name("consumer_" + suffix);
        inputs.push_back(input);
        results.push_back(std::make_shared<ov::op::v0::Result>(consumer));
        built.eliminated.push_back(eliminated);
    }
    built.model = std::make_shared<ov::Model>(results, inputs);
    return built;
}

void write_shape_consumer_plan(const std::filesystem::path& plan_path,
                               const std::string& shape_family,
                               const std::string& consumer_family) {
    std::ofstream plan(plan_path);
    ASSERT_TRUE(plan.is_open());
    auto write_group = [&plan](const std::string& layer, const std::string& family) {
        plan << "<group gflops=\"0\"";
        if (!family.empty()) {
            plan << " repeated=\"" << family << "\"";
        }
        plan << "><input name=\"" << layer << "\"/><output name=\"" << layer << "\"/><layer name=\"" << layer
             << "\"/></group>";
    };
    auto write_block = [&plan](const std::string& family, const std::string& layer_stem) {
        if (family.empty()) {
            return;
        }
        plan << "<block id=\"" << family << "\"><match>";
        for (unsigned i = 0; i < 3; ++i) {
            plan << "<layer name=\"" << layer_stem << i << "\"/>";
        }
        plan << "</match></block>";
    };
    plan << "<ensemble gflops=\"0\"><partitioning>";
    for (unsigned i = 0; i < 3; ++i) {
        write_group("gather_" + std::to_string(i), shape_family);
    }
    for (unsigned i = 0; i < 3; ++i) {
        write_group("consumer_" + std::to_string(i), consumer_family);
    }
    plan << "</partitioning><repeated>";
    write_block(shape_family, "gather_");
    write_block(consumer_family, "consumer_");
    plan << "</repeated></ensemble>";
    ASSERT_TRUE(plan.good());
}

// Evaluates a subgraph (or the function it calls) with every input set to `input_value`.
int64_t evaluate_subgraph(const ov::npuw::Partitioning& partitioning,
                          const ov::npuw::Subgraph& subgraph,
                          int64_t input_value) {
    auto model = subgraph._funcall.empty() ? std::make_shared<ov::Model>(subgraph._results, subgraph._parameters)
                                           : partitioning.functions.at(subgraph._funcall)._model;
    ov::TensorVector inputs;
    for (const auto& input : model->inputs()) {
        inputs.emplace_back(input.get_element_type(), input.get_shape());
        std::fill_n(inputs.back().data<int64_t>(), inputs.back().get_size(), input_value);
    }
    ov::TensorVector outputs{ov::Tensor(ov::element::i64, ov::Shape{1})};
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs.front().data<int64_t>()[0];
}

template <typename Param>
class EliminatedSubgraphTestBase : public ::testing::TestWithParam<Param> {
protected:
    const std::filesystem::path plan_path = make_unique_temp_path("npuw_eliminated_subgraphs", ".xml");

    ~EliminatedSubgraphTestBase() override {
        std::error_code ec;
        std::filesystem::remove(plan_path, ec);  // never throws from a destructor
    }
};

class EliminatedSubgraphTest : public EliminatedSubgraphTestBase<std::tuple<bool, bool, unsigned>> {};

TEST_P(EliminatedSubgraphTest, ExcludesEliminatedInstancesFromFunctions) {
    const auto [force_funcall, cwai, eliminated_mask] = GetParam();
    const auto built = build_shape_consumer_model(eliminated_mask);
    write_shape_consumer_plan(plan_path, force_funcall ? "" : "shape_family", "");

    auto cfg = make_cfg({{"NPUW_PLAN", plan_path.string()},
                         {"NPUW_FUNCALL_FOR_ALL", force_funcall ? "YES" : "NO"},
                         {"NPUW_FOLD", cwai ? "NO" : "YES"},
                         {"NPUW_CWAI", cwai ? "YES" : "NO"}});
    auto partitioning = ov::npuw::getPartitioning(built.model, cfg);

    ASSERT_EQ(partitioning.subgraphs.size(), 6u);
    std::set<std::string> live_shape_functions;
    unsigned live_shapes = 0;
    for (unsigned i = 0; i < 3; ++i) {
        const bool eliminated = built.eliminated[i];
        const auto& subgraph = partitioning.subgraphs[i];
        EXPECT_EQ(subgraph._optimized_out, eliminated);
        if (eliminated) {
            EXPECT_TRUE(subgraph._funcall.empty());
            EXPECT_TRUE(subgraph._repeated_id.empty());
            EXPECT_TRUE(subgraph._results.empty());
        } else {
            ++live_shapes;
            EXPECT_FALSE(subgraph._funcall.empty());
            live_shape_functions.insert(subgraph._funcall);
            ASSERT_EQ(partitioning.functions.count(subgraph._funcall), 1u);
            const auto& function = partitioning.functions.at(subgraph._funcall);
            ASSERT_EQ(function._model->get_results().size(), 1u);
            EXPECT_EQ(function._model->output().get_element_type(), ov::element::i64);
            EXPECT_EQ(function._model->output().get_shape(), ov::Shape{1});
            ASSERT_EQ(function._model->inputs().size(), 1u);
            EXPECT_EQ(evaluate_subgraph(partitioning, subgraph, 5), 5);
        }
        const auto& consumer = partitioning.subgraphs[i + 3];
        EXPECT_FALSE(consumer._optimized_out);
        auto consumer_model = consumer._funcall.empty()
                                  ? std::make_shared<ov::Model>(consumer._results, consumer._parameters)
                                  : partitioning.functions.at(consumer._funcall)._model;
        ASSERT_EQ(consumer_model->inputs().size(), eliminated ? 0u : 1u);
        EXPECT_EQ(evaluate_subgraph(partitioning, consumer, 5), eliminated ? 3 : 6);
    }
    EXPECT_EQ(live_shape_functions.size(), !force_funcall && !cwai && live_shapes ? 1u : live_shapes);
    const auto expected_functions = live_shape_functions.size() + (force_funcall ? 3u : 0u);
    EXPECT_EQ(partitioning.functions.size(), expected_functions);
    ASSERT_EQ(partitioning.input_to_prev_output.size(), live_shapes);
    for (const auto& link : partitioning.input_to_prev_output) {
        EXPECT_FALSE(partitioning.subgraphs.at(link.first.first)._optimized_out);
        EXPECT_FALSE(partitioning.subgraphs.at(link.second.first)._optimized_out);
    }
    for (const auto& entry : partitioning.functions) {
        EXPECT_FALSE(entry.second._model->get_results().empty());
    }
}

INSTANTIATE_TEST_SUITE_P(Partitioning,
                         EliminatedSubgraphTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Range(0u, 8u)));

class EliminatedProducerRepeatedConsumerTest : public EliminatedSubgraphTestBase<unsigned> {};

// The consumers of the (partially) eliminated shape family form a repeated block of
// their own. Where the producer is eliminated, the consumer's Parameter is folded into a
// Constant. The consumers stay a function only if all or none of the producers are
// eliminated; otherwise their instances differ and are kept as plain subgraphs.
TEST_P(EliminatedProducerRepeatedConsumerTest, FoldsConsumersOfEliminatedProducers) {
    const auto eliminated_mask = GetParam();
    const auto built = build_shape_consumer_model(eliminated_mask);
    write_shape_consumer_plan(plan_path, "shape_family", "consumer_family");

    auto cfg = make_cfg({{"NPUW_PLAN", plan_path.string()}, {"NPUW_FOLD", "YES"}});
    auto partitioning = ov::npuw::getPartitioning(built.model, cfg);

    const bool uniform = eliminated_mask == 0u || eliminated_mask == 7u;
    ASSERT_EQ(partitioning.subgraphs.size(), 6u);
    std::set<std::string> consumer_functions;
    for (unsigned i = 0; i < 3; ++i) {
        const bool eliminated = built.eliminated[i];
        EXPECT_EQ(partitioning.subgraphs[i]._optimized_out, eliminated);
        const auto& consumer = partitioning.subgraphs[i + 3];
        EXPECT_FALSE(consumer._optimized_out);
        // function calls carry the function in _funcall (their _repeated_id is moved there),
        // demoted instances have neither
        EXPECT_EQ(consumer._funcall.empty(), !uniform) << "consumer_" << i;
        EXPECT_TRUE(consumer._repeated_id.empty()) << "consumer_" << i;
        if (!consumer._funcall.empty()) {
            consumer_functions.insert(consumer._funcall);
            const auto& function = partitioning.functions.at(consumer._funcall);
            EXPECT_EQ(function._model->inputs().size(), eliminated ? 0u : 1u);
        }
        EXPECT_EQ(evaluate_subgraph(partitioning, consumer, 5), eliminated ? 3 : 6) << "consumer_" << i;
    }
    EXPECT_EQ(consumer_functions.size(), uniform ? 1u : 0u);
    const bool any_live_shape = std::count(built.eliminated.begin(), built.eliminated.end(), false) > 0;
    EXPECT_EQ(partitioning.functions.size(), consumer_functions.size() + (any_live_shape ? 1u : 0u));
}

INSTANTIATE_TEST_SUITE_P(Partitioning, EliminatedProducerRepeatedConsumerTest, ::testing::Range(0u, 8u));

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

TEST(PartitioningOptionsTest, OnlyAttnIsolatedFamiliesIgnoreKeepBlockSizeThreshold) {
    // With KEEP_BLOCK_SIZE=2 and KEEP_BLOCK_TAG=attn, only the attn-tagged
    // single-node repeated family bypasses the size threshold. Non-attn
    // isolated single-node families are left unfrozen and fuse into adjacent remnants.
    constexpr std::size_t N = 30;
    auto ext_cfg = abc_attn_base_cfg;
    ext_cfg["NPUW_ONLINE_KEEP_BLOCK_SIZE"] = "2";
    ext_cfg["NPUW_ONLINE_KEEP_BLOCKS_TAGGED"] = "attn";
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

// Same three op families, but all of them share one isolate tag, so producer and
// consumer families can actually be merged into a single repeating block.
static const ::intel_npu::Config::ConfigMap attn_only_base_cfg = {
    {"NPUW_ONLINE_PIPELINE", "REP"},
    {"NPUW_ONLINE_ISOLATE", "Op:Relu/attn,Op:Sigmoid/attn,Op:Tanh/attn"},
    {"NPUW_ONLINE_KEEP_BLOCKS", "5"},
    {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
};

TEST(PartitioningOptionsTest, KeepBlocksTaggedMergesFewerOccurrencesThanKeepBlocks) {
    // 4 repetitions produce only 3 mergeable producer/consumer pairs - less than
    // KEEP_BLOCKS=5. Without NPUW_ONLINE_KEEP_BLOCKS_TAGGED the count floor rejects
    // the merge and the model stays at one group per op.
    constexpr std::size_t N = 4;
    auto model = build_abc_attn_model(N);

    auto plain_cfg = make_cfg(attn_only_base_cfg);
    auto plain_ens = ov::npuw::online::buildPartitioning(model, plain_cfg);

    EXPECT_TRUE(plain_ens.repeated.empty());
    EXPECT_EQ(partition_boundaries(plain_ens),
              (std::set<std::set<std::string>>{{"relu_0", "sigmoid_0"},
                                               {"relu_1", "tanh_0"},
                                               {"sigmoid_1"},
                                               {"tanh_1"},
                                               {"relu_2"},
                                               {"sigmoid_2"},
                                               {"tanh_2"},
                                               {"relu_3"},
                                               {"sigmoid_3"},
                                               {"tanh_3"}}));

    auto tagged_map = attn_only_base_cfg;
    tagged_map["NPUW_ONLINE_KEEP_BLOCKS_TAGGED"] = "attn";
    auto tagged_cfg = make_cfg(tagged_map);
    auto tagged_ens = ov::npuw::online::buildPartitioning(model, tagged_cfg);

    // With the tag, the floor is bypassed and the 3 pairs are coalesced into a
    // repeating block of 3 layers, leaving only the chain head and tail behind.
    EXPECT_EQ(partition_boundaries(tagged_ens),
              (std::set<std::set<std::string>>{{"relu_0"},
                                               {"sigmoid_0"},
                                               {"relu_1", "sigmoid_1", "tanh_0"},
                                               {"relu_2", "sigmoid_2", "tanh_1"},
                                               {"relu_3", "sigmoid_3", "tanh_2"},
                                               {"tanh_3"}}));

    // matches[i] is the set of instances of the i-th layer of the block, so a
    // 3-layer block repeated 3 times is expected here.
    const auto three_layer_block =
        std::find_if(tagged_ens.repeated.begin(), tagged_ens.repeated.end(), [](const auto& rep) {
            return rep.second.matches.size() == 3u;
        });
    ASSERT_NE(three_layer_block, tagged_ens.repeated.end());
    EXPECT_EQ(three_layer_block->second.matches.front().size(), 3u);
}

TEST(PartitioningOptionsTest, KeepBlocksTaggedDoesNotCoalesceSingleOccurrencePairs) {
    // Only relu_0 -> sigmoid_0 is a mergeable pair here; relu_1 and sigmoid_1 are
    // separated by tanh_0. Merging that lone pair would split both two-instance
    // families and leave degenerate one-instance repeated blocks, so the
    // "at least two instances" guard stays in force even for a keep tag.
    auto model = build_single_pair_model();

    auto tagged_map = attn_only_base_cfg;
    tagged_map["NPUW_ONLINE_KEEP_BLOCKS_TAGGED"] = "attn";
    auto tagged_cfg = make_cfg(tagged_map);
    auto tagged_ens = ov::npuw::online::buildPartitioning(model, tagged_cfg);

    EXPECT_EQ(partition_boundaries(tagged_ens),
              (std::set<std::set<std::string>>{{"relu_0"},
                                               {"sigmoid_0"},
                                               {"abs_0", "ceil_0"},
                                               {"floor_0", "sqrt_0"},
                                               {"exp_0", "log_0"},
                                               {"sin_0", "cos_0"},
                                               {"neg_0"},
                                               {"relu_1"},
                                               {"tanh_0"},
                                               {"sigmoid_1"}}));

    // The two tagged families are kept as-is: single-layer blocks with two instances each.
    EXPECT_EQ(tagged_ens.repeated.size(), 2u);
    for (const auto& rep : tagged_ens.repeated) {
        EXPECT_EQ(rep.second.matches.size(), 1u);
        EXPECT_EQ(rep.second.matches.front().size(), 2u);
    }
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
