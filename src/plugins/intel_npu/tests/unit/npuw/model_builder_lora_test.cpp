// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "llm_test_helpers.hpp"
#include "openvino/op/parameter.hpp"

namespace {

std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& m, const std::string& needle) {
    for (const auto& p : m->get_parameters()) {
        if (p->get_friendly_name().find(needle) != std::string::npos) {
            return p;
        }
    }
    return nullptr;
}

bool has_param(const std::shared_ptr<ov::Model>& m, const std::string& needle) {
    return find_param(m, needle) != nullptr;
}

TEST(ModelBuilderLoraTest, StatelessAdapterExposesLoraParameters) {
    auto model = ov::test::npuw::build_lora_adapter_test_model();
    ASSERT_TRUE(model);
    EXPECT_TRUE(has_param(model, "lora_state_"));
    EXPECT_TRUE(model->get_sinks().empty());

    // NPUW hardcodes the low-rank dim per tensor name: A=[r,in], B=[out,r], alpha=[1,r] f32.
    const auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::LoRAConfig>();
    auto a = find_param(model, "q_proj.MatMul.A");
    auto b = find_param(model, "q_proj.MatMul.B");
    auto alpha = find_param(model, "q_proj.MatMul.alpha");
    ASSERT_TRUE(a && b && alpha);
    EXPECT_EQ(a->get_shape(), ov::Shape({cfg.lora_rank, cfg.hidden_size}));
    EXPECT_EQ(b->get_shape(), ov::Shape({cfg.hidden_size, cfg.lora_rank}));
    EXPECT_EQ(alpha->get_shape(), ov::Shape({1, cfg.lora_rank}));
    EXPECT_EQ(alpha->get_element_type(), ov::element::f32);
}

TEST(ModelBuilderLoraTest, TargetsFilterLimitsAdaptedProjections) {
    ov::test::npuw::ModelBuilder builder;
    auto config = ov::test::npuw::make_test_model_config<ov::test::npuw::LoRAConfig>();
    config.lora_targets = {"q_proj", "v_proj"};

    auto model = builder.build_lora_adapter(config);
    ASSERT_TRUE(model);
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.A"));
    EXPECT_TRUE(has_param(model, "v_proj.MatMul.A"));
    EXPECT_FALSE(has_param(model, "k_proj.MatMul.A"));
    EXPECT_FALSE(has_param(model, "gate_proj.MatMul.A"));
}

TEST(ModelBuilderLoraTest, StatefulAdapterExposesLoraStates) {
    ov::test::npuw::ModelBuilder builder;
    auto config = ov::test::npuw::make_test_model_config<ov::test::npuw::LoRAConfig>();
    config.lora_stateful = true;

    auto model = builder.build_lora_adapter(config);
    ASSERT_TRUE(model);
    EXPECT_FALSE(model->get_sinks().empty());

    bool has_lora_variable = false;
    for (const auto& variable : model->get_variables()) {
        const auto& id = variable->get_info().variable_id;
        if (id.find("lora_state_") == 0 && id.find(".MatMul.") != std::string::npos) {
            has_lora_variable = true;
            break;
        }
    }
    EXPECT_TRUE(has_lora_variable);
}

TEST(ModelBuilderLoraTest, BuildLlmInjectsLoraIntoAttentionAndFfn) {
    auto model = ov::test::npuw::build_lora_llm_test_model();
    ASSERT_TRUE(model);
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.A"));
    EXPECT_TRUE(has_param(model, "down_proj.MatMul.A"));
}

TEST(ModelBuilderLoraTest, BuildLlmWithoutLoraHasNoLoraParameters) {
    auto model = ov::test::npuw::build_llm_test_model();
    ASSERT_TRUE(model);
    EXPECT_FALSE(has_param(model, "lora_state_"));
}

}  // namespace
