// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
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

TEST(PartitioningEffectTest, PipelineNoneMergesUnaryModelIntoSingleGroup) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_EQ(ens.groups.size(), 1u);
}

TEST(PartitioningEffectTest, AvoidsOnNonePipelineSplitUnaryModel) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_AVOID", "Op:Sin/NPU,Op:Cos/NPU"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_EQ(ens.groups.size(), 3u);
}

TEST(PartitioningEffectTest, IsolateOptionTagsUnaryGroups) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "REP"}, {"NPUW_ONLINE_ISOLATE", "Op:Sin/compute"}});
    auto ens = ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);
    EXPECT_TRUE(std::any_of(ens.groups.begin(), ens.groups.end(), [](const ov::npuw::Group& group) {
        return group.gettag() == "compute";
    }));
}

TEST(PartitioningEffectTest, DumpPlanWritesXmlFile) {
    const auto dump_path = std::filesystem::temp_directory_path() / "npuw_partitioning_effect_dump.xml";
    std::filesystem::remove(dump_path);

    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_ONLINE_DUMP_PLAN", dump_path.string()}});
    (void)ov::npuw::online::buildPartitioning(build_unary_chain_model(), cfg);

    ASSERT_TRUE(std::filesystem::exists(dump_path));
    std::ifstream stream(dump_path);
    std::string xml((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    EXPECT_NE(xml.find("<ensemble"), std::string::npos);

    std::filesystem::remove(dump_path);
}

TEST(PartitioningEffectTest, ComputePipelineMarksComputeGroupsAsNoFold) {
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

TEST(PartitioningEffectTest, SpatialPipelineDoesNotAnnotateFullPrefillModelWithoutSpatialRange) {
    auto cfg = make_cfg({{"NPUW_ONLINE_PIPELINE", "SPATIAL"}, {"NPUW_SPATIAL_NWAY", "16"}});
    auto partitioning = ov::npuw::getPartitioning(build_static_prefill_model(), cfg);

    bool seen_spatial = false;
    for (const auto& [_, function] : partitioning.functions) {
        seen_spatial |= function._spatial.has_value();
    }
    EXPECT_FALSE(seen_spatial);
}

TEST(PartitioningEffectTest, DynamicAttentionRejectsUnisolatedPrefillGraph) {
    auto model = build_static_prefill_model();
    const auto attention = ov::npuw::function::Attention::from(model);

    EXPECT_FALSE(attention.has_value());
}

TEST(PartitioningEffectTest, PyramidAttentionRejectsUnisolatedGenerateGraph) {
    auto model = build_static_generate_model();
    const auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    EXPECT_FALSE(pyramid.has_value());
}

}  // namespace
