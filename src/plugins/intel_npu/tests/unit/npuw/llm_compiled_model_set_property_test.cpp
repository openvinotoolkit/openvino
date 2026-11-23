// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "llm_test_helpers.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
using ov::test::npuw::MockSubCompiledModel;
using ov::test::npuw::NullPlugin;

// Stage model which records every set_property call it receives
class PropertyRecordingSubCompiledModel final : public MockSubCompiledModel {
public:
    using MockSubCompiledModel::MockSubCompiledModel;

    void set_property(const ov::AnyMap& properties) override {
        if (reject) {
            OPENVINO_THROW("Rejected: property can't be changed");
        }
        set_calls.push_back(properties);
    }

    bool reject = false;
    std::vector<ov::AnyMap> set_calls;
};

class LLMCompiledModelSetPropertyTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    // Prefill, two generate variants (pyramid) and the shared LM head: every kind of
    // inner model LLMCompiledModel keeps. m_kvcache_compiled aliases the last variant.
    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model() {
        ov::AnyMap props = {{"NPUW_LLM", "YES"},
                            {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                            {"NPUW_LLM_MIN_RESPONSE_LEN", "128"},
                            {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                            {"NPUW_LLM_SHARED_HEAD", "YES"}};
        auto factory = [this](const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<const ov::IPlugin>& plugin,
                              const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            auto stage = std::make_shared<PropertyRecordingSubCompiledModel>(model, plugin, props);
            m_stages.push_back(stage);
            return stage;
        };
        return std::make_unique<ov::npuw::LLMCompiledModel>(ov::test::npuw::build_llm_test_model(),
                                                            m_plugin,
                                                            props,
                                                            factory);
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
    std::vector<std::shared_ptr<PropertyRecordingSubCompiledModel>> m_stages;
};

TEST_F(LLMCompiledModelSetPropertyTest, ModelPriorityIsForwardedOnceToEveryStage) {
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model());
    ASSERT_NE(compiled, nullptr);
    // prefill + 2 generate variants + lm head
    ASSERT_EQ(m_stages.size(), 4u);

    const ov::AnyMap priority = {{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}};
    ASSERT_NO_THROW(compiled->set_property(priority));

    for (std::size_t i = 0; i < m_stages.size(); ++i) {
        ASSERT_EQ(m_stages[i]->set_calls.size(), 1u) << "Stage #" << i;
        const auto& props = m_stages[i]->set_calls.front();
        ASSERT_EQ(props.size(), 1u);
        EXPECT_EQ(props.at(ov::hint::model_priority.name()).as<ov::hint::Priority>(), ov::hint::Priority::HIGH);
    }
}

TEST_F(LLMCompiledModelSetPropertyTest, StageErrorIsPropagated) {
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model());
    ASSERT_NE(compiled, nullptr);
    ASSERT_FALSE(m_stages.empty());

    // Stages validate which keys can be changed; a rejection must reach the caller
    for (const auto& stage : m_stages) {
        stage->reject = true;
    }
    EXPECT_ANY_THROW(compiled->set_property({{"NPUW_FOLD", true}}));
}

}  // namespace
