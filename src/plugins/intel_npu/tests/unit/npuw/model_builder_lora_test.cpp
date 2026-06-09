// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "llm_test_helpers.hpp"

namespace {

bool has_param(const std::shared_ptr<ov::Model>& m, const std::string& needle) {
    for (const auto& p : m->get_parameters()) {
        if (p->get_friendly_name().find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

TEST(ModelBuilderLoraTest, StatelessAdapterExposesLoraParameters) {
    auto model = ov::test::npuw::build_lora_adapter_test_model();
    ASSERT_TRUE(model);
    EXPECT_TRUE(has_param(model, "lora_state_"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.A"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.B"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.alpha"));
    EXPECT_TRUE(model->get_sinks().empty());
}

TEST(ModelBuilderLoraTest, StatefulAdapterExposesLoraStates) {
    ov::test::npuw::ModelBuilder builder;
    auto config = ov::test::npuw::make_test_model_config<ov::test::npuw::LoRAConfig>();
    config.lora_stateful = true;

    auto model = builder.build_lora_adapter(config);
    ASSERT_TRUE(model);
    EXPECT_FALSE(model->get_sinks().empty());
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
