// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiled_model.hpp"
#include "llm_compiled_model.hpp"
#include "model_builder.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "serialization.hpp"
#include "weights_bank.hpp"

namespace {

class NullPlugin : public ov::IPlugin {
public:
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                      const ov::AnyMap&,
                                                      const ov::SoPtr<ov::IRemoteContext>&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::AnyMap&) const override {
        return {};
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&,
                                                     const ov::SoPtr<ov::IRemoteContext>&,
                                                     const ov::AnyMap&) const override {
        return {};
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&, const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        return {};
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        return {};
    }
};

class MockSubCompiledModel : public ov::npuw::ICompiledModel_v0 {
public:
    MockSubCompiledModel(const std::shared_ptr<ov::Model>& model,
                         const std::shared_ptr<const ov::IPlugin>& plugin,
                         const ov::AnyMap&)
        : ov::npuw::ICompiledModel_v0(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return {};
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return {};
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        return 0;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}
};

struct CompileCall {
    std::string                friendly_name;
    ov::AnyMap                 props;
    std::shared_ptr<ov::Model> model;
};

class RecordingFactory {
public:
    ov::npuw::LLMCompiledModel::CompiledModelFactory make_factory() {
        return [this](const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::AnyMap& props) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
            m_calls.push_back({model->get_friendly_name(), props, model});
            return std::make_shared<MockSubCompiledModel>(model, plugin, props);
        };
    }

    const std::vector<CompileCall>& calls() const {
        return m_calls;
    }

    std::size_t count_suffix(std::string_view suffix) const {
        return std::count_if(m_calls.begin(), m_calls.end(), [suffix](const CompileCall& call) {
            return call.friendly_name.size() >= suffix.size() &&
                   call.friendly_name.compare(call.friendly_name.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
    }

    const CompileCall* find_suffix(std::string_view suffix) const {
        const auto it = std::find_if(m_calls.begin(), m_calls.end(), [suffix](const CompileCall& call) {
            return call.friendly_name.size() >= suffix.size() &&
                   call.friendly_name.compare(call.friendly_name.size() - suffix.size(), suffix.size(), suffix) == 0;
        });
        return it == m_calls.end() ? nullptr : &(*it);
    }

private:
    std::vector<CompileCall> m_calls;
};

class LLMCompiledModelFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    static ov::test::npuw::ModelConfig base_model_config() {
        ov::test::npuw::ModelConfig cfg;
        cfg.num_layers = 2;
        cfg.hidden_size = 64;
        cfg.num_heads = 4;
        cfg.head_dim = 16;
        cfg.num_kv_heads = 4;
        cfg.vocab_size = 256;
        return cfg;
    }

    std::shared_ptr<ov::Model> build_llm_model() const {
        ov::test::npuw::ModelBuilder mb;
        return mb.build_model(base_model_config());
    }

    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"}, {"NPUW_LLM_MAX_PROMPT_LEN", "128"}, {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    }

    static void merge_props(ov::AnyMap& dst, const ov::AnyMap& src) {
        for (const auto& [key, value] : src) {
            dst[key] = value;
        }
    }

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const std::shared_ptr<ov::Model>& model,
                                                                      const ov::AnyMap& extra_props,
                                                                      RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(model, m_plugin, props, recorder.make_factory());
    }

    static const CompileCall& require_call(const RecordingFactory& recorder, std::string_view suffix) {
        const auto* call = recorder.find_suffix(suffix);
        OPENVINO_ASSERT(call != nullptr, "Missing compile call with suffix: ", std::string(suffix));
        return *call;
    }

    static std::string prop_string(const ov::AnyMap& props, const std::string& key) {
        const auto it = props.find(key);
        OPENVINO_ASSERT(it != props.end(), "Missing property: ", key);
        return it->second.as<std::string>();
    }

    static void expect_prop(const ov::AnyMap& props, const std::string& key, const std::string& expected) {
        EXPECT_EQ(prop_string(props, key), expected);
    }

    static void expect_missing_prop(const ov::AnyMap& props, const std::string& key) {
        EXPECT_EQ(props.count(key), 0u) << "Unexpected property present: " << key;
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(LLMCompiledModelFactoryTest, TwoModelPipeline) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(recorder.calls().size(), 2u);
    EXPECT_EQ(recorder.count_suffix("_kv192"), 1u);
    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.find_suffix("_lm_head"), nullptr);
}

TEST_F(LLMCompiledModelFactoryTest, ThreeModelPipeline) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(recorder.calls().size(), 3u);
    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.count_suffix("_lm_head"), 1u);
}

TEST_F(LLMCompiledModelFactoryTest, GeneratePyramidCompilesMultipleGenerateVariants) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {{"NPUW_LLM_SHARED_HEAD", "NO"},
                        {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                        {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                        {"NPUW_LLM_MIN_RESPONSE_LEN", "128"}};

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_GT(recorder.calls().size(), 2u);
    EXPECT_GT(std::count_if(recorder.calls().begin(), recorder.calls().end(), [](const CompileCall& call) {
                  return call.friendly_name.find("_kv") != std::string::npos;
              }),
              1);
    EXPECT_TRUE(compiled->get_property("NPUW_LLM_GENERATE_PYRAMID").as<bool>());
}

TEST_F(LLMCompiledModelFactoryTest, ConfigOverridesAndAdditionsArePassedToFactory) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {
        {"NPUW_LLM_SHARED_HEAD", "YES"},
        {"NPUW_DEVICES", "CPU"},
        {"NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"PREFILL_ONLY", "base"}, {"NPUW_ONLINE_PIPELINE", "NONE"}}},
        {"++NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"PREFILL_ONLY", "override"}, {"PREFILL_EXTRA", "1"}}},
        {"NPUW_LLM_GENERATE_CONFIG", ov::AnyMap{{"GENERATE_ONLY", "base"}, {"NPUW_ONLINE_PIPELINE", "NONE"}}},
        {"++NPUW_LLM_GENERATE_CONFIG", ov::AnyMap{{"GENERATE_ONLY", "override"}, {"GENERATE_EXTRA", "1"}}},
        {"NPUW_LLM_SHARED_HEAD_CONFIG", ov::AnyMap{{"HEAD_ONLY", "base"}}},
        {"++NPUW_LLM_SHARED_HEAD_CONFIG", ov::AnyMap{{"HEAD_ONLY", "override"}, {"HEAD_EXTRA", "1"}}},
    };

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    expect_prop(prefill.props, "PREFILL_ONLY", "override");
    expect_prop(prefill.props, "PREFILL_EXTRA", "1");
    expect_prop(prefill.props, "NPUW_DEVICES", "CPU");

    const auto& generate = require_call(recorder, "_kv192");
    expect_prop(generate.props, "GENERATE_ONLY", "override");
    expect_prop(generate.props, "GENERATE_EXTRA", "1");
    expect_prop(generate.props, "NPUW_DEVICES", "CPU");

    const auto& head = require_call(recorder, "_lm_head");
    expect_prop(head.props, "HEAD_ONLY", "override");
    expect_prop(head.props, "HEAD_EXTRA", "1");
}

TEST_F(LLMCompiledModelFactoryTest, AttentionHintsPropagateToStageConfigs) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {{"NPUW_LLM_SHARED_HEAD", "NO"},
                        {"NPUW_LLM_PREFILL_ATTENTION_HINT", "PYRAMID"},
                        {"NPUW_LLM_GENERATE_ATTENTION_HINT", "DYNAMIC"}};

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    expect_prop(prefill.props, "NPUW_ATTN", "PYRAMID");
    expect_prop(prefill.props, "NPUW_ONLINE_PIPELINE", "REP");
    expect_prop(prefill.props, "NPUW_ONLINE_ISOLATE", "ATTN");
    expect_prop(prefill.props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
    expect_prop(prefill.props, "NPUW_UNFOLD_IREQS", "NO");
    expect_prop(prefill.props, "NPUW_FALLBACK_EXEC", "NO");

    const auto& generate = require_call(recorder, "_kv192");
    expect_prop(generate.props, "NPUW_ATTN", "DYNAMIC");
    expect_prop(generate.props, "NPUW_ONLINE_PIPELINE", "REP");
    expect_prop(generate.props, "NPUW_ONLINE_ISOLATE", "ATTN");
    expect_prop(generate.props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
    expect_prop(generate.props, "NPUW_UNFOLD_IREQS", "NO");
    expect_prop(generate.props, "NPUW_FALLBACK_EXEC", "NO");
}

TEST_F(LLMCompiledModelFactoryTest, VisibleLlmPropertiesRoundTripThroughCompiledModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {{"NPUW_LLM_MAX_PROMPT_LEN", "160"},
                        {"NPUW_LLM_MIN_RESPONSE_LEN", "96"},
                        {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                        {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                        {"NPUW_LLM_GENERATE_HINT", "BEST_PERF"},
                        {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                        {"NPUW_LLM_GENERATE_ATTENTION_HINT", "HFA"},
                        {"NPUW_LLM_SHARED_HEAD", "NO"},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE", "0"}};

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    EXPECT_TRUE(compiled->get_property("NPUW_LLM").as<bool>());
    EXPECT_EQ(compiled->get_property("NPUW_LLM_MAX_PROMPT_LEN").as<uint32_t>(), 160u);
    EXPECT_EQ(compiled->get_property("NPUW_LLM_MIN_RESPONSE_LEN").as<uint32_t>(), 96u);
    EXPECT_TRUE(compiled->get_property("NPUW_LLM_GENERATE_PYRAMID").as<bool>());
    EXPECT_EQ(compiled->get_property("NPUW_LLM_PREFILL_HINT").as<std::string>(), "STATIC");
    EXPECT_EQ(compiled->get_property("NPUW_LLM_GENERATE_HINT").as<std::string>(), "BEST_PERF");
    EXPECT_EQ(compiled->get_property("NPUW_LLM_PREFILL_ATTENTION_HINT").as<std::string>(), "STATIC");
    EXPECT_EQ(compiled->get_property("NPUW_LLM_GENERATE_ATTENTION_HINT").as<std::string>(), "HFA");
    EXPECT_FALSE(compiled->get_property("NPUW_LLM_SHARED_HEAD").as<bool>());
    EXPECT_EQ(compiled->get_property("NPUW_LLM_PREFILL_CHUNK_SIZE").as<uint64_t>(), 0u);
}

}  // namespace
