// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiled_model.hpp"
#include "llm_compiled_model.hpp"
#include "model_builder.hpp"
#include "openvino/op/slice.hpp"
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

class LLMCompiledModelGraphEffectTest : public ::testing::Test {
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

    std::shared_ptr<ov::Model> build_model() const {
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

    std::unique_ptr<ov::npuw::LLMCompiledModel> create_compiled_model(const ov::AnyMap& extra_props,
                                                                      RecordingFactory& recorder) const {
        auto props = base_props();
        merge_props(props, extra_props);
        return std::make_unique<ov::npuw::LLMCompiledModel>(build_model(), m_plugin, props, recorder.make_factory());
    }

    template <class Port>
    static bool port_has_name(const Port& port, std::string_view needle) {
        const auto& names = port.get_names();
        return std::any_of(names.begin(), names.end(), [needle](const std::string& name) {
            return name.find(needle) != std::string::npos;
        });
    }

    static std::optional<ov::Output<const ov::Node>> find_input(const std::shared_ptr<ov::Model>& model,
                                                                std::string_view needle) {
        const auto inputs = model->inputs();
        const auto it = std::find_if(inputs.begin(), inputs.end(), [needle](const auto& input) {
            return port_has_name(input, needle);
        });
        if (it == inputs.end()) {
            return std::nullopt;
        }
        return *it;
    }

    static std::optional<ov::Output<const ov::Node>> find_output(const std::shared_ptr<ov::Model>& model,
                                                                 std::string_view needle) {
        const auto outputs = model->outputs();
        const auto it = std::find_if(outputs.begin(), outputs.end(), [needle](const auto& output) {
            return port_has_name(output, needle);
        });
        if (it == outputs.end()) {
            return std::nullopt;
        }
        return *it;
    }

    static std::size_t count_inputs(const std::shared_ptr<ov::Model>& model, std::string_view needle) {
        const auto inputs = model->inputs();
        return std::count_if(inputs.begin(), inputs.end(), [needle](const auto& input) {
            return port_has_name(input, needle);
        });
    }

    static std::size_t count_outputs(const std::shared_ptr<ov::Model>& model, std::string_view needle) {
        const auto outputs = model->outputs();
        return std::count_if(outputs.begin(), outputs.end(), [needle](const auto& output) {
            return port_has_name(output, needle);
        });
    }

    template <class Op>
    static std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
        const auto ops = model->get_ops();
        return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
            return ov::is_type<Op>(op);
        });
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(LLMCompiledModelGraphEffectTest, SharedHeadAddsHeadModelAndSlicesPrefillEmbeds) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* lm_head = recorder.find_suffix("_lm_head");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(lm_head, nullptr);

    const auto embeds = find_output(prefill->model, ov::npuw::LLMCompiledModel::output_embeds);
    ASSERT_TRUE(embeds.has_value());
    ASSERT_TRUE(embeds->get_partial_shape().is_static());
    EXPECT_EQ(embeds->get_shape(), (ov::Shape{1, 8, 64}));

    const auto head_input = lm_head->model->input(0);
    ASSERT_TRUE(head_input.get_partial_shape().is_static());
    EXPECT_EQ(head_input.get_shape(), (ov::Shape{1, 8, 64}));
    EXPECT_GE(count_ops<ov::op::v8::Slice>(prefill->model), 1u);
}

TEST_F(LLMCompiledModelGraphEffectTest, PromptResponseAndGenerationLengthsDriveStaticShapes) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto* generate = recorder.find_suffix("_kv192");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(generate, nullptr);

    const auto prefill_ids = find_input(prefill->model, "input_ids");
    const auto prefill_mask = find_input(prefill->model, "attention_mask");
    const auto generate_ids = find_input(generate->model, "input_ids");
    const auto generate_mask = find_input(generate->model, "attention_mask");
    ASSERT_TRUE(prefill_ids.has_value());
    ASSERT_TRUE(prefill_mask.has_value());
    ASSERT_TRUE(generate_ids.has_value());
    ASSERT_TRUE(generate_mask.has_value());
    EXPECT_EQ(prefill_ids->get_shape(), (ov::Shape{1, 128}));
    EXPECT_EQ(prefill_mask->get_shape(), (ov::Shape{1, 128}));
    EXPECT_EQ(generate_ids->get_shape(), (ov::Shape{1, 8}));
    EXPECT_EQ(generate_mask->get_shape(), (ov::Shape{1, 192}));
}

TEST_F(LLMCompiledModelGraphEffectTest, DynamicChunkPrefillKeepsPastKvInputsAndExportsPresentKvOutputs) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_GT(count_inputs(prefill->model, "past_key_values"), 0u);
    EXPECT_GT(count_outputs(prefill->model, "present"), 0u);
    const auto ids = find_input(prefill->model, "input_ids");
    ASSERT_TRUE(ids.has_value());
    EXPECT_EQ(ids->get_shape(), (ov::Shape{1, 32}));
}

TEST_F(LLMCompiledModelGraphEffectTest, StaticPrefillRemovesPastKvInputsAndKeepsPresentKvOutputs) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(count_inputs(prefill->model, "past_key_values"), 0u);
    EXPECT_GT(count_outputs(prefill->model, "present"), 0u);
    const auto ids = find_input(prefill->model, "input_ids");
    ASSERT_TRUE(ids.has_value());
    EXPECT_EQ(ids->get_shape(), (ov::Shape{1, 128}));
}

TEST_F(LLMCompiledModelGraphEffectTest, GeneratePyramidCreatesIntermediateKvVariants) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    std::vector<std::string> kv_names;
    for (const auto& call : recorder.calls()) {
        if (call.friendly_name.find("_kv") != std::string::npos) {
            kv_names.push_back(call.friendly_name);
        }
    }
    EXPECT_THAT(kv_names, ::testing::UnorderedElementsAre(::testing::EndsWith("_kv1152"),
                                                          ::testing::EndsWith("_kv2176")));
}

}  // namespace
