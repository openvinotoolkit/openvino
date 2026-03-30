// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llm_test_helpers.hpp"
#include "whisper/prepare_whisper_model.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {
using ov::test::npuw::CompileCall;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::RecordingFactory;

class LLMCompiledModelFactoryOptionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    std::shared_ptr<ov::Model> build_llm_model() const {
        return ov::test::npuw::build_llm_test_model();
    }

    std::shared_ptr<ov::Model> build_whisper_decoder_model() const {
        return ov::test::npuw::build_whisper_decoder_test_model();
    }

    std::shared_ptr<ov::Model> build_embedding_model() const {
        return ov::test::npuw::build_embedding_test_model();
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

    static const CompileCall& require_call_containing(const RecordingFactory& recorder, std::string_view fragment) {
        const auto* call = recorder.find_contains(fragment);
        OPENVINO_ASSERT(call != nullptr, "Missing compile call containing: ", std::string(fragment));
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

    static bool has_input_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
        const auto inputs = model->inputs();
        return std::any_of(inputs.begin(), inputs.end(), [&name](const auto& input) {
            const auto& names = input.get_names();
            return std::any_of(names.begin(), names.end(), [&name](const auto& candidate) {
                return candidate.find(name) != std::string::npos;
            });
        });
    }

    static bool has_output_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
        const auto outputs = model->outputs();
        return std::any_of(outputs.begin(), outputs.end(), [&name](const auto& output) {
            const auto& names = output.get_names();
            return std::any_of(names.begin(), names.end(), [&name](const auto& candidate) {
                return candidate.find(name) != std::string::npos;
            });
        });
    }

    std::shared_ptr<ov::IPlugin> m_plugin;
};

TEST_F(LLMCompiledModelFactoryOptionsTest, SharedHeadDisabledBuildsPrefillAndGenerateStagesOnly) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(recorder.calls().size(), 2u);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.find_suffix("_lm_head"), nullptr);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, SharedHeadEnabledBuildsPrefillGenerateAndHeadStages) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(recorder.calls().size(), 3u);
    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.count_suffix("_lm_head"), 1u);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, GeneratePyramidBuildsExpectedGenerateStageCount) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {{"NPUW_LLM_SHARED_HEAD", "NO"},
                        {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                        {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                        {"NPUW_LLM_MIN_RESPONSE_LEN", "128"}};

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(recorder.calls().size(), 3u);
    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.count_contains("_kv"), 2u);
    EXPECT_TRUE(compiled->get_property("NPUW_LLM_GENERATE_PYRAMID").as<bool>());
}

TEST_F(LLMCompiledModelFactoryOptionsTest, ConfigOverridesAndAdditionsArePassedToFactory) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {
        {"NPUW_LLM_SHARED_HEAD", "YES"},
        {"NPUW_DEVICES", "CPU"},
        {"NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_DEVICES", "CPU"}}},
        {"++NPUW_LLM_PREFILL_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "CPU,NPU"}, {"NPUW_UNFOLD_IREQS", "YES"}}},
        {"NPUW_LLM_GENERATE_CONFIG", ov::AnyMap{{"NPUW_ONLINE_PIPELINE", "NONE"}, {"NPUW_FALLBACK_EXEC", "YES"}}},
        {"++NPUW_LLM_GENERATE_CONFIG", ov::AnyMap{{"NPUW_FALLBACK_EXEC", "NO"}, {"NPUW_FUNCALL_ASYNC", "NO"}}},
        {"NPUW_LLM_SHARED_HEAD_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "CPU"}}},
        {"++NPUW_LLM_SHARED_HEAD_CONFIG", ov::AnyMap{{"NPUW_DEVICES", "NPU"}}},
    };

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    expect_prop(prefill.props, "NPUW_ONLINE_PIPELINE", "NONE");
    expect_prop(prefill.props, "NPUW_DEVICES", "CPU,NPU");
    expect_prop(prefill.props, "NPUW_UNFOLD_IREQS", "YES");

    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_ONLINE_PIPELINE", "NONE");
    expect_prop(generate.props, "NPUW_FALLBACK_EXEC", "NO");
    expect_prop(generate.props, "NPUW_FUNCALL_ASYNC", "NO");

    const auto& head = require_call(recorder, "_lm_head");
    expect_prop(head.props, "NPUW_DEVICES", "NPU");
}

TEST_F(LLMCompiledModelFactoryOptionsTest, AttentionHintsPropagateToStageConfigs) {
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

    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_ATTN", "DYNAMIC");
    expect_prop(generate.props, "NPUW_ONLINE_PIPELINE", "REP");
    expect_prop(generate.props, "NPUW_ONLINE_ISOLATE", "ATTN");
    expect_prop(generate.props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
    expect_prop(generate.props, "NPUW_UNFOLD_IREQS", "NO");
    expect_prop(generate.props, "NPUW_FALLBACK_EXEC", "NO");
}

TEST_F(LLMCompiledModelFactoryOptionsTest, VisibleLlmPropertiesRoundTripThroughCompiledModel) {
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

TEST_F(LLMCompiledModelFactoryOptionsTest, DefaultStageConfigsCarryBaselineNpuwOptions) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        expect_prop(call->props, "NPU_USE_NPUW", "YES");
        expect_prop(call->props, "NPUW_DEVICES", "NPU");
        expect_prop(call->props, "NPUW_FOLD", "YES");
        expect_prop(call->props, "NPUW_DCOFF_TYPE", "f16");
        expect_prop(call->props, "NPUW_DCOFF_SCALE", "YES");
        EXPECT_FALSE(prop_string(call->props, "NPUW_WEIGHTS_BANK").empty());
    }

    expect_prop(prefill.props, "NPUW_SLICE_OUT", "YES");
    expect_prop(prefill.props, "NPUW_FUNCALL_ASYNC", "YES");

    expect_missing_prop(generate.props, "NPUW_SLICE_OUT");
    expect_prop(generate.props, "NPUW_FUNCALL_ASYNC", "YES");

    expect_missing_prop(head.props, "NPUW_SLICE_OUT");
    expect_missing_prop(head.props, "NPUW_FUNCALL_ASYNC");
    expect_prop(head.props, "NPUW_ONLINE_PIPELINE", "NONE");

    EXPECT_EQ(prop_string(prefill.props, "NPUW_WEIGHTS_BANK"), prop_string(generate.props, "NPUW_WEIGHTS_BANK"));
    EXPECT_EQ(prop_string(prefill.props, "NPUW_WEIGHTS_BANK"), prop_string(head.props, "NPUW_WEIGHTS_BANK"));
}

TEST_F(LLMCompiledModelFactoryOptionsTest, CommonRuntimeAndDebugOptionsForwardToAllStages) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap props = {
        {"NPUW_LLM_SHARED_HEAD", "YES"},
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_DEVICES", "CPU,NPU"},
        {"NPUW_SUBMODEL_DEVICE", "0:CPU,last:NPU"},
        {"NPUW_WEIGHTS_BANK_ALLOC", "CPU"},
        {"NPUW_CACHE_DIR", "/tmp/npuw-cache"},
        {"NPUW_PARALLEL_COMPILE", "YES"},
        {"NPUW_FUNCALL_ASYNC", "NO"},
        {"NPUW_UNFOLD_IREQS", "YES"},
        {"NPUW_FALLBACK_EXEC", "YES"},
        {"NPUW_ACC_CHECK", "YES"},
        {"NPUW_ACC_THRESH", "0.25"},
        {"NPUW_ACC_DEVICE", "CPU"},
        {"NPUW_DUMP_FULL", "YES"},
        {"NPUW_DUMP_SUBS", "YES"},
        {"NPUW_DUMP_SUBS_DIR", "/tmp/npuw-dumps"},
        {"NPUW_DUMP_SUBS_ON_FAIL", "last"},
        {"NPUW_DUMP_IO", "0,last"},
        {"NPUW_DUMP_IO_ITERS", "YES"},
    };

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    const std::array<const CompileCall*, 3> calls = {&require_call(recorder, "_prefill"),
                                                     &require_call_containing(recorder, "_kv"),
                                                     &require_call(recorder, "_lm_head")};
    for (const auto* call : calls) {
        expect_prop(call->props, "NPU_USE_NPUW", "YES");
        expect_prop(call->props, "NPUW_DEVICES", "CPU,NPU");
        expect_prop(call->props, "NPUW_SUBMODEL_DEVICE", "0:CPU,last:NPU");
        expect_prop(call->props, "NPUW_WEIGHTS_BANK_ALLOC", "CPU");
        expect_prop(call->props, "NPUW_CACHE_DIR", "/tmp/npuw-cache");
        expect_prop(call->props, "NPUW_PARALLEL_COMPILE", "YES");
        expect_prop(call->props, "NPUW_FUNCALL_ASYNC", "NO");
        expect_prop(call->props, "NPUW_UNFOLD_IREQS", "YES");
        expect_prop(call->props, "NPUW_FALLBACK_EXEC", "YES");
        expect_prop(call->props, "NPUW_ACC_CHECK", "YES");
        expect_prop(call->props, "NPUW_ACC_THRESH", "0.25");
        expect_prop(call->props, "NPUW_ACC_DEVICE", "CPU");
        expect_prop(call->props, "NPUW_DUMP_FULL", "YES");
        expect_prop(call->props, "NPUW_DUMP_SUBS", "YES");
        expect_prop(call->props, "NPUW_DUMP_SUBS_DIR", "/tmp/npuw-dumps");
        expect_prop(call->props, "NPUW_DUMP_SUBS_ON_FAIL", "last");
        expect_prop(call->props, "NPUW_DUMP_IO", "0,last");
        expect_prop(call->props, "NPUW_DUMP_IO_ITERS", "YES");
    }
}

TEST_F(LLMCompiledModelFactoryOptionsTest, FastCompileGenerateHintKeepsUnfoldedGenerateDefaults) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_HINT", "FAST_COMPILE"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_UNFOLD_IREQS", "YES");
    expect_missing_prop(generate.props, "NPUW_SLICE_OUT");
}

TEST_F(LLMCompiledModelFactoryOptionsTest, BestPerfGenerateHintForcesStandaloneGeneratePartitioning) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_HINT", "BEST_PERF"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_ONLINE_PIPELINE", "NONE");
    expect_missing_prop(generate.props, "NPUW_SLICE_OUT");
}

TEST_F(LLMCompiledModelFactoryOptionsTest, StaticAttentionHintsAvoidAttentionIsolationPipelineOverrides) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_PREFILL_ATTENTION_HINT", "STATIC"},
                                                      {"NPUW_LLM_GENERATE_ATTENTION_HINT", "STATIC"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(prefill.props, "NPUW_ATTN", "STATIC");
    expect_prop(generate.props, "NPUW_ATTN", "STATIC");
    // Static attention should leave partitioning defaults unchanged rather than forcing attention isolation.
    EXPECT_EQ(prefill.props.count("NPUW_ONLINE_PIPELINE"), 0u);
    EXPECT_EQ(generate.props.count("NPUW_ONLINE_PIPELINE"), 0u);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, HfaAttentionHintsEnableAttentionIsolationSettings) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_PREFILL_ATTENTION_HINT", "HFA"},
                                                      {"NPUW_LLM_GENERATE_ATTENTION_HINT", "HFA"},
                                                      {"NPUW_ATTN_HFA_FUSED", "YES"},
                                                      {"NPUW_ATTN_DYN", "YES"},
                                                      {"NPUW_ATTN_NO_COPY", "YES"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    for (const auto* call : {&prefill, &generate}) {
        expect_prop(call->props, "NPUW_ATTN", "HFA");
        expect_prop(call->props, "NPUW_ONLINE_PIPELINE", "REP");
        expect_prop(call->props, "NPUW_ONLINE_ISOLATE", "ATTN");
        expect_prop(call->props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
        expect_prop(call->props, "NPUW_UNFOLD_IREQS", "NO");
        expect_prop(call->props, "NPUW_ATTN_HFA_FUSED", "YES");
        expect_prop(call->props, "NPUW_ATTN_DYN", "YES");
        expect_prop(call->props, "NPUW_ATTN_NO_COPY", "YES");
    }
}

TEST_F(LLMCompiledModelFactoryOptionsTest, CacheRopeEnabledRoundsTripThroughCompiledModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_CACHE_ROPE", "YES"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_TRUE(compiled->get_property("NPUW_LLM_CACHE_ROPE").as<bool>());
    EXPECT_NE(recorder.find_suffix("_prefill"), nullptr);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, CacheRopeDisabledRoundsTripThroughCompiledModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_CACHE_ROPE", "NO"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_FALSE(compiled->get_property("NPUW_LLM_CACHE_ROPE").as<bool>());
    EXPECT_NE(recorder.find_suffix("_prefill"), nullptr);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, WhisperOptionCompilesSyntheticDecoderModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_whisper_decoder_model(),
                                                     {{"NPUW_WHISPER", "YES"}, {"NPUW_WHISPER_EOS_TOKEN", "42"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_GE(recorder.calls().size(), 2u);
    EXPECT_NE(recorder.find_suffix("_prefill"), nullptr);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, WhisperPreparationAddsKvCacheInputsAndPresentOutputs) {
    auto model = build_whisper_decoder_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    const auto input_count_before = model->inputs().size();
    const auto output_count_before = model->outputs().size();

    EXPECT_TRUE(ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(model));
    auto prepared = model;

    EXPECT_GT(prepared->inputs().size(), input_count_before);
    EXPECT_EQ(prepared->outputs().size(), output_count_before);
    EXPECT_TRUE(has_input_name(prepared, "attention_mask"));
    EXPECT_TRUE(has_input_name(prepared, "cache_position"));
    EXPECT_TRUE(has_input_name(prepared, "past_key_values"));
    EXPECT_TRUE(has_output_name(prepared, "present"));
    EXPECT_EQ(prepared->input("input_ids").get_partial_shape()[1].get_length(), 1);
}

TEST_F(LLMCompiledModelFactoryOptionsTest, WhisperPrefillPreparationAddsCrossAttentionMaskWithoutCachePosition) {
    auto model = build_whisper_decoder_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    auto lhs_seq_size = static_cast<uint32_t>(
        model->input("encoder_hidden_states").get_partial_shape()[1].get_length());
    EXPECT_TRUE(ov::npuw::util::PrepareWhisperPrefillModel(128, lhs_seq_size).run_on_model(model));
    auto prepared = model;

    EXPECT_TRUE(has_input_name(prepared, "attention_mask"));
    EXPECT_FALSE(has_input_name(prepared, "cache_position"));
    EXPECT_TRUE(has_output_name(prepared, "present"));
}

}  // namespace
