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

#include "llm_test_helpers.hpp"
#include "openvino/op/slice.hpp"

namespace {
using ov::test::npuw::CompileCall;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::RecordingFactory;

class LLMCompiledModelGraphOptionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_plugin = std::make_shared<NullPlugin>();
    }

    std::shared_ptr<ov::Model> build_model() const {
        return ov::test::npuw::build_llm_test_model();
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

    static const CompileCall& require_call_containing(const RecordingFactory& recorder, std::string_view fragment) {
        const auto* call = recorder.find_contains(fragment);
        OPENVINO_ASSERT(call != nullptr, "Missing compile call containing: ", std::string(fragment));
        return *call;
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

TEST_F(LLMCompiledModelGraphOptionsTest, SharedHeadAddsHeadModelAndSlicesPrefillEmbeds) {
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

    const auto embeds = find_output(prefill->model, ov::npuw::LLMCompiledModel::layer_names::output_embeds);
    ASSERT_TRUE(embeds.has_value());
    ASSERT_TRUE(embeds->get_partial_shape().is_static());
    EXPECT_EQ(embeds->get_shape(), (ov::Shape{1, 8, 64}));

    const auto head_input = lm_head->model->input(0);
    ASSERT_TRUE(head_input.get_partial_shape().is_static());
    EXPECT_EQ(head_input.get_shape(), (ov::Shape{1, 8, 64}));
    EXPECT_GE(count_ops<ov::op::v8::Slice>(prefill->model), 1u);
}

TEST_F(LLMCompiledModelGraphOptionsTest, PromptResponseAndGenerationLengthsDriveStaticShapes) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto* prefill = recorder.find_suffix("_prefill");
    const auto& generate = require_call_containing(recorder, "_kv");
    ASSERT_NE(prefill, nullptr);

    const auto prefill_ids = find_input(prefill->model, "input_ids");
    const auto prefill_mask = find_input(prefill->model, "attention_mask");
    const auto generate_ids = find_input(generate.model, "input_ids");
    const auto generate_mask = find_input(generate.model, "attention_mask");
    ASSERT_TRUE(prefill_ids.has_value());
    ASSERT_TRUE(prefill_mask.has_value());
    ASSERT_TRUE(generate_ids.has_value());
    ASSERT_TRUE(generate_mask.has_value());
    EXPECT_EQ(prefill_ids->get_shape(), (ov::Shape{1, 128}));
    EXPECT_EQ(prefill_mask->get_shape(), (ov::Shape{1, 128}));
    EXPECT_EQ(generate_ids->get_shape(), (ov::Shape{1, 8}));
    EXPECT_EQ(generate_mask->get_shape(), (ov::Shape{1, 192}));
}

TEST_F(LLMCompiledModelGraphOptionsTest, DynamicChunkPrefillKeepsPastKvInputsAndExportsPresentKvOutputs) {
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

TEST_F(LLMCompiledModelGraphOptionsTest, StaticPrefillRemovesPastKvInputsAndKeepsPresentKvOutputs) {
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

TEST_F(LLMCompiledModelGraphOptionsTest, GeneratePyramidBuildsTwoStaticGenerateVariants) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_GENERATE_PYRAMID", "YES"},
                                                      {"NPUW_LLM_MAX_PROMPT_LEN", "2048"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    std::vector<ov::Shape> generate_attention_mask_shapes;
    for (const auto& call : recorder.calls()) {
        if (call.friendly_name.find("_kv") != std::string::npos) {
            const auto mask = find_input(call.model, "attention_mask");
            ASSERT_TRUE(mask.has_value());
            ASSERT_TRUE(mask->get_partial_shape().is_static());
            generate_attention_mask_shapes.push_back(mask->get_shape());
        }
    }
    EXPECT_EQ(generate_attention_mask_shapes.size(), 2u);
    EXPECT_THAT(generate_attention_mask_shapes,
                ::testing::UnorderedElementsAre(ov::Shape({1, 1152}), ov::Shape({1, 2176})));
}

}  // namespace
