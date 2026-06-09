// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "model_builder.hpp"

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
    ov::test::npuw::ModelBuilder builder;
    ov::test::npuw::LoRAConfig config;
    config.num_layers = 1;
    config.hidden_size = 8;
    config.vocab_size = 32;
    config.precision = ov::element::f16;
    config.lora_targets = {"q_proj", "v_proj"};

    auto model = builder.build_lora_adapter(config);
    ASSERT_TRUE(model);
    EXPECT_TRUE(has_param(model, "lora_state_"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.A"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.B"));
    EXPECT_TRUE(has_param(model, "q_proj.MatMul.alpha"));
    EXPECT_TRUE(model->get_sinks().empty());
}

TEST(ModelBuilderLoraTest, StatefulAdapterExposesLoraStates) {
    ov::test::npuw::ModelBuilder builder;
    ov::test::npuw::LoRAConfig config;
    config.num_layers = 1;
    config.hidden_size = 8;
    config.vocab_size = 32;
    config.precision = ov::element::f16;
    config.lora_targets = {"q_proj"};
    config.lora_stateful = true;

    auto model = builder.build_lora_adapter(config);
    ASSERT_TRUE(model);
    EXPECT_FALSE(model->get_sinks().empty());
}

}  // namespace
