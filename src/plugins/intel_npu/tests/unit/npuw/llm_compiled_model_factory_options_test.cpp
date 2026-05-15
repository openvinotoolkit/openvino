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
#include "openvino/runtime/intel_npu/properties.hpp"
#include "whisper/prepare_whisper_model.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

namespace {
using ov::test::npuw::CompileCall;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::RecordingFactory;

class ArchAwarePlugin final : public NullPlugin {
public:
    ArchAwarePlugin(std::string arch, int64_t max_tiles) : m_arch(std::move(arch)), m_max_tiles(max_tiles) {}

    ov::Any get_property(const std::string& name, const ov::AnyMap&) const override {
        if (name == ov::device::architecture.name()) {
            return m_arch;
        }
        if (name == ov::intel_npu::max_tiles.name()) {
            return m_max_tiles;
        }
        if (name == ov::intel_npu::compiler_version.name()) {
            return static_cast<int64_t>(0);
        }
        if (name == ov::supported_properties.name()) {
            return std::vector<ov::PropertyName>{};
        }
        return {};
    }

private:
    std::string m_arch;
    int64_t m_max_tiles;
};

std::shared_ptr<ov::MockICore> attach_mock_core_with_npu_device(const std::shared_ptr<ov::IPlugin>& plugin, 
    std::vector<std::string> device_list = std::vector<std::string>{"0"}) {
    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    plugin->set_core(core);

    ON_CALL(*core,
            get_property(testing::StrEq("NPU"),
                         testing::StrEq(ov::available_devices.name()),
                         testing::An<const ov::AnyMap&>()))
        .WillByDefault([device_list](const std::string&, const std::string&, const ov::AnyMap&) -> ov::Any {
                return ov::Any(device_list);
            });
    return core;
}

class LLMCompiledModelFactoryOptionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    std::shared_ptr<ov::Model> build_llm_model() const {
        return ov::test::npuw::build_llm_test_model();
    }

    std::shared_ptr<ov::Model> build_moe_llm_model() const {
        return ov::test::npuw::build_moe_llm_test_model();
    }

    std::shared_ptr<ov::Model> build_whisper_decoder_model() const {
        return ov::test::npuw::build_whisper_decoder_test_model();
    }

    std::shared_ptr<ov::Model> build_embedding_model() const {
        return ov::test::npuw::build_embedding_test_model();
    }

    std::shared_ptr<ov::Model> build_embedding_decoder_model() const {
        return ov::test::npuw::build_embedding_decoder_test_model();
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

    static int64_t prop_i64(const ov::AnyMap& props, const std::string& key) {
        const auto it = props.find(key);
        OPENVINO_ASSERT(it != props.end(), "Missing property: ", key);
        return it->second.as<int64_t>();
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
                        {"NPUW_LLM_GENERATE_ATTENTION_HINT", "DYNAMIC"},
                        {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                        {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"}};

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

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"}},
                                                     recorder));
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
    expect_prop(prefill.props, "NPUW_ATTN", "PYRAMID");
    expect_prop(prefill.props, "NPUW_ONLINE_PIPELINE", "REP");
    expect_prop(prefill.props, "NPUW_ONLINE_ISOLATE", "ATTN");
    expect_prop(prefill.props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
    expect_prop(prefill.props, "NPUW_UNFOLD_IREQS", "NO");
    expect_missing_prop(prefill.props, "NPUW_FALLBACK_EXEC");

    expect_missing_prop(generate.props, "NPUW_SLICE_OUT");
    expect_prop(generate.props, "NPUW_FUNCALL_ASYNC", "YES");
    expect_missing_prop(generate.props, "NPUW_ATTN");

    expect_missing_prop(head.props, "NPUW_SLICE_OUT");
    expect_missing_prop(head.props, "NPUW_FUNCALL_ASYNC");
    expect_prop(head.props, "NPUW_ONLINE_PIPELINE", "NONE");

    EXPECT_EQ(prop_string(prefill.props, "NPUW_WEIGHTS_BANK"), prop_string(generate.props, "NPUW_WEIGHTS_BANK"));
    EXPECT_EQ(prop_string(prefill.props, "NPUW_WEIGHTS_BANK"), prop_string(head.props, "NPUW_WEIGHTS_BANK"));
}

TEST(NPUWAttentionHintOptionDefaultsTest, PrefillAndGenerateAttentionHintsHaveIndependentDefaults) {
    EXPECT_EQ(::intel_npu::NPUW_LLM_PREFILL_ATTENTION_HINT::defaultValue(),
              ::intel_npu::npuw::llm::AttentionHint::PYRAMID);
    EXPECT_EQ(::intel_npu::NPUW_LLM_GENERATE_ATTENTION_HINT::defaultValue(),
              ::intel_npu::npuw::llm::AttentionHint::STATIC);
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

TEST_F(LLMCompiledModelFactoryOptionsTest, ArchIn5000SeriesSetsNpuTilesInDefaultStageConfigs) {
    m_plugin = std::make_shared<ArchAwarePlugin>("5010", 3);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        EXPECT_EQ(prop_i64(call->props, "NPU_TILES"), 3);
        expect_prop(call->props,
                    "NPU_COMPILATION_MODE_PARAMS",
                    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm");
    }
}

struct UserStageConfigParam {
    std::string arch;
    int64_t max_tiles;              // reported by the plugin (arch default)
    int64_t user_tiles;             // NPU_TILES value explicitly provided by the user
    // Note: NPU_COMPILATION_MODE_PARAMS is a space-separated token string.
    // Supplying it here performs a *full replace* of the arch-computed value
    // (which may include tokens like optimization-level=3 or performance-hint-override=latency).
    // The production code emits a LOG_WARN when the user value differs from the arch-computed one
    // so the user is informed that arch-specific tokens will be lost unless explicitly repeated.
    // A proper additive mechanism (analogous to ++NPUW_LLM_PREFILL_CONFIG) does not yet exist
    // for this property.
    std::string user_compile_params;  // NPU_COMPILATION_MODE_PARAMS override; empty = not provided
};

class StageConfigUserOverrideTest : public LLMCompiledModelFactoryOptionsTest,
                                    public ::testing::WithParamInterface<UserStageConfigParam> {};

// User-provided NPU_TILES (and optionally NPU_COMPILATION_MODE_PARAMS) must take priority
// over the arch-computed defaults from set_max_tiles_based_on_arch, regardless of the NPU platform.
TEST_P(StageConfigUserOverrideTest, UserSettingsOverrideArchDefaults) {
    const auto& param = GetParam();
    m_plugin = std::make_shared<ArchAwarePlugin>(param.arch, param.max_tiles);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ov::AnyMap extra_props = {{"NPUW_LLM_SHARED_HEAD", "YES"}, {"NPU_TILES", param.user_tiles}};
    if (!param.user_compile_params.empty()) {
        extra_props["NPU_COMPILATION_MODE_PARAMS"] = param.user_compile_params;
    }

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), extra_props, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        EXPECT_EQ(call->props.count("NPU_TILES"), 1u) << "NPU_TILES must appear exactly once in compiled stage config";
        EXPECT_EQ(prop_i64(call->props, "NPU_TILES"), param.user_tiles);

        if (!param.user_compile_params.empty()) {
            EXPECT_EQ(call->props.count("NPU_COMPILATION_MODE_PARAMS"), 1u)
                << "NPU_COMPILATION_MODE_PARAMS must appear exactly once in compiled stage config";
            // Full replace: the user value entirely replaces the arch-computed string.
            expect_prop(call->props, "NPU_COMPILATION_MODE_PARAMS", param.user_compile_params);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllArchVariants,
    StageConfigUserOverrideTest,
    ::testing::Values(
        // NPU 2700: arch default sets no tiles; user overrides NPU_TILES with 1 and 2
        UserStageConfigParam{"2700", 2, 1, ""},
        UserStageConfigParam{"2700", 2, 2, ""},
        // NPU 4000: arch default is max_tiles=4 + optimization-level=3; user fully replaces compile params
        UserStageConfigParam{"4000", 4, 1, ""},
        UserStageConfigParam{"4000", 4, 2, "compute-layers-with-higher-precision=Sqrt,Power"},
        // NPU 5010: arch default is max_tiles=3; user overrides NPU_TILES and compile params
        UserStageConfigParam{"5010", 3, 1, ""},
        UserStageConfigParam{"5010", 3, 2, "compute-layers-with-higher-precision=Sqrt,Power"},
        // NPU 5020: arch default is max_tiles=3; user overrides NPU_TILES and compile params
        UserStageConfigParam{"5020", 3, 1, ""},
        UserStageConfigParam{"5020", 3, 2, "compute-layers-with-higher-precision=Sqrt,Power"},
        // AUTO_DETECT: arch default appends performance-hint-override=latency; user fully replaces
        UserStageConfigParam{"AUTO_DETECT", 4, 1, ""},
        UserStageConfigParam{"AUTO_DETECT", 4, 2, "compute-layers-with-higher-precision=Sqrt,Power"}),
    [](const ::testing::TestParamInfo<UserStageConfigParam>& info) {
        std::string name = info.param.arch;
        std::replace(name.begin(), name.end(), '_', 'x');
        name += "_tiles" + std::to_string(info.param.user_tiles);
        if (!info.param.user_compile_params.empty()) {
            name += "_with_compile_params";
        }
        return name;
    });

TEST_F(LLMCompiledModelFactoryOptionsTest, Arch2700SkipsNpuTilesInDefaultStageConfigs) {
    m_plugin = std::make_shared<ArchAwarePlugin>("2700", 2);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        expect_missing_prop(call->props, "NPU_TILES");
        expect_prop(call->props,
                    "NPU_COMPILATION_MODE_PARAMS",
                    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm");
    }
}
// avtual config for one of platform with embargo details.
TEST_F(LLMCompiledModelFactoryOptionsTest, ArchAutoDetectSkipsNpuTilesInDefaultStageConfigs) {
    m_plugin = std::make_shared<ArchAwarePlugin>("AUTO_DETECT", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        expect_missing_prop(call->props, "NPU_TILES");
        expect_prop(call->props,
                    "NPU_COMPILATION_MODE_PARAMS",
                    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm performance-hint-override=latency");
    }
}

TEST_F(LLMCompiledModelFactoryOptionsTest, Arch4000AddsOptimizationLevelAndSetsNpuTiles) {
    m_plugin = std::make_shared<ArchAwarePlugin>("4000", 4);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        EXPECT_EQ(prop_i64(call->props, "NPU_TILES"), 4);
        expect_prop(call->props,
                    "NPU_COMPILATION_MODE_PARAMS",
                    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm optimization-level=3");
    }
}

TEST_F(LLMCompiledModelFactoryOptionsTest, ArchAtLeast6000UsesDefaultConfig) {
    m_plugin = std::make_shared<ArchAwarePlugin>("7000", 8);
    const auto core = attach_mock_core_with_npu_device(m_plugin);

    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(), {{"NPUW_LLM_SHARED_HEAD", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    const auto& head = require_call(recorder, "_lm_head");

    for (const auto* call : {&prefill, &generate, &head}) {
        expect_missing_prop(call->props, "NPU_TILES");
        expect_prop(call->props,
                    "NPU_COMPILATION_MODE_PARAMS",
                    "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm");
    }
}


TEST_F(LLMCompiledModelFactoryOptionsTest, FastCompileGenerateHintKeepsCurrentGenerateDefaults) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_HINT", "FAST_COMPILE"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);
    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_UNFOLD_IREQS", "YES");
    expect_prop(generate.props, "NPUW_DQ", "YES");
    expect_missing_prop(generate.props, "NPUW_SLICE_OUT");
}

TEST_F(LLMCompiledModelFactoryOptionsTest, MissingNpuBackendKeepsCurrentGenerateDefaults) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    auto core = std::make_shared<testing::NiceMock<ov::MockICore>>();
    m_plugin->set_core(core);
    ON_CALL(*core, get_property(testing::StrEq("NPU"), testing::StrEq(ov::available_devices.name()), testing::_))
        .WillByDefault([](const std::string&, const std::string&, const ov::AnyMap&) -> ov::Any {
            OPENVINO_THROW("No available backend");
        });

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_HINT", "FAST_COMPILE"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_call_containing(recorder, "_kv");
    expect_prop(generate.props, "NPUW_UNFOLD_IREQS", "YES");
    expect_prop(generate.props, "NPUW_DQ", "YES");
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
    expect_missing_prop(prefill.props, "NPUW_ATTN");
    expect_missing_prop(generate.props, "NPUW_ATTN");
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
                                                      {"NPUW_ATTN_NO_COPY", "YES"},
                                                      {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"}},
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

TEST_F(LLMCompiledModelFactoryOptionsTest, MoeModelKeepsHostRoutedStageIntegration) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_moe_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_PREFILL_MOE_HINT", "HOST_ROUTED"},
                                                      {"NPUW_LLM_GENERATE_MOE_HINT", "HOST_ROUTED"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_call(recorder, "_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");

    for (const auto* call : {&prefill, &generate}) {
        expect_prop(call->props, "NPUW_ONLINE_PIPELINE", "REP");
        expect_prop(call->props, "NPUW_ONLINE_ISOLATE", "MOE");
        expect_prop(call->props, "NPUW_ONLINE_KEEP_BLOCK_SIZE", "4");
        expect_prop(call->props, "NPUW_UNFOLD_IREQS", "NO");
    }

    EXPECT_EQ(recorder.count_suffix("_prefill"), 1u);
    EXPECT_EQ(recorder.count_contains("_kv"), 1u);
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

// RopeCache replaces Sin/Cos with a Gather-from-LUT. When enabled the prefill
// sub-model must have no Sin/Cos nodes left.
TEST_F(LLMCompiledModelFactoryOptionsTest, CacheRopeEnabledRemovesSinCosFromPrefill) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_CACHE_ROPE", "YES"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);

    const auto& ops = prefill->model->get_ops();
    auto sin_count = std::count_if(ops.begin(), ops.end(),
                                   [](const auto& op) { return ov::is_type<ov::op::v0::Sin>(op); });
    auto cos_count = std::count_if(ops.begin(), ops.end(),
                                   [](const auto& op) { return ov::is_type<ov::op::v0::Cos>(op); });
    EXPECT_EQ(sin_count, 0) << "RopeCache should have replaced all Sin nodes in the prefill model";
    EXPECT_EQ(cos_count, 0) << "RopeCache should have replaced all Cos nodes in the prefill model";
}

// When rope caching is disabled the prefill model must still contain Sin/Cos
// (i.e. the RoPE pattern is present but untransformed).
TEST_F(LLMCompiledModelFactoryOptionsTest, CacheRopeDisabledKeepsSinCosInPrefill) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_llm_model(),
                                                     {{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_CACHE_ROPE", "NO"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);

    const auto& ops = prefill->model->get_ops();
    auto sin_count = std::count_if(ops.begin(), ops.end(),
                                   [](const auto& op) { return ov::is_type<ov::op::v0::Sin>(op); });
    auto cos_count = std::count_if(ops.begin(), ops.end(),
                                   [](const auto& op) { return ov::is_type<ov::op::v0::Cos>(op); });
    EXPECT_GT(sin_count, 0) << "Sin nodes must remain when rope caching is disabled";
    EXPECT_GT(cos_count, 0) << "Cos nodes must remain when rope caching is disabled";
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

    EXPECT_TRUE(ov::npuw::util::PrepareWhisperPrefillModel(
                    128, static_cast<uint32_t>(ov::test::npuw::WhisperConfig{}.max_source_positions),
                    false /*decompose_sdpa*/)
                    .run_on_model(model));
    auto prepared = model;

    EXPECT_TRUE(has_input_name(prepared, "attention_mask"));
    EXPECT_FALSE(has_input_name(prepared, "cache_position"));
    EXPECT_TRUE(has_output_name(prepared, "present"));
}

TEST_F(LLMCompiledModelFactoryOptionsTest, TextEmbedOptionCompilesEmbeddingDecoderModel) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_embedding_decoder_model(),
                                                      {{"NPUW_TEXT_EMBED", "YES"},
                                                       {"NPUW_LLM_SHARED_HEAD", "NO"}},
                                                      recorder));
    ASSERT_NE(compiled, nullptr);
    EXPECT_GE(recorder.calls().size(), 1u);
    EXPECT_NE(recorder.find_suffix("_prefill"), nullptr);
}

}  // namespace
