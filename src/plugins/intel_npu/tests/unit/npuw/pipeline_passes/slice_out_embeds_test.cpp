// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/result.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class SliceOutEmbedsPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    static bool output_embeds_has_gather_producer(const std::shared_ptr<ov::Model>& model) {
        for (const auto& output : model->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find(ov::npuw::LLMCompiledModel::layer_names::output_embeds) != std::string::npos) {
                    // output is an Output<const Node> over the Result node
                    auto result_node = output.get_node_shared_ptr();
                    if (!result_node || result_node->inputs().empty())
                        return false;
                    auto producer = result_node->input(0).get_source_output().get_node_shared_ptr();
                    return ov::is_type<ov::op::v8::Gather>(producer);
                }
            }
        }
        return false;
    }
};

TEST_F(SliceOutEmbedsPassTest, SharedHeadUsesGatherWithInternalIndices) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_PROPAGATE_SLICE_UP", "NO"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                                     recorder));

    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& lm_head = require_sub_model(recorder, "_lm_head");
    (void)lm_head;  // asserted to exist; contents checked by reshape_sliced_head tests

    // output_embeds must exist on the prefill model
    const auto embeds = find_output(prefill.model, ov::npuw::LLMCompiledModel::layer_names::output_embeds);
    ASSERT_TRUE(embeds.has_value()) << "output_embeds output not found on prefill model";

    // Shape must be static and trimmed to generation token length
    ASSERT_TRUE(embeds->get_partial_shape().is_static())
        << "output_embeds shape is not static: " << embeds->get_partial_shape();
    EXPECT_EQ(embeds->get_shape(), (ov::Shape{1, 8, 64}));

    EXPECT_TRUE(output_embeds_has_gather_producer(prefill.model));
    EXPECT_TRUE(find_input(prefill.model, ov::npuw::util::kLogitsSelectionIndicesParamName).has_value());
}

TEST_F(SliceOutEmbedsPassTest, SharedHeadUsesGatherWhenGenerationTokenLenEqualsPromptLen) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // MAX_GENERATION_TOKEN_LEN=128 equals MAX_PROMPT_LEN=128, so no slicing is needed.
    ASSERT_NO_THROW(compiled = create_compiled_model(
                        {{"NPUW_LLM_SHARED_HEAD", "YES"}, {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "128"}},
                        recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");

    // output_embeds must still be exported
    const auto embeds = find_output(prefill.model, ov::npuw::LLMCompiledModel::layer_names::output_embeds);
    ASSERT_TRUE(embeds.has_value()) << "output_embeds output not found on prefill model";

    // Shape should be the full prompt-length shape, unmodified
    ASSERT_TRUE(embeds->get_partial_shape().is_static())
        << "output_embeds shape is not static: " << embeds->get_partial_shape();
    EXPECT_EQ(embeds->get_shape(), (ov::Shape{1, 128, 64}));

    EXPECT_TRUE(output_embeds_has_gather_producer(prefill.model));
    EXPECT_TRUE(find_input(prefill.model, ov::npuw::util::kLogitsSelectionIndicesParamName).has_value());
}

TEST_F(SliceOutEmbedsPassTest, ChunkedPrefillUsesGraphLastTokenSelection) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "1"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& lm_head = require_sub_model(recorder, "_lm_head");
    const auto embeds = find_output(prefill.model, ov::npuw::LLMCompiledModel::layer_names::output_embeds);
    ASSERT_TRUE(embeds.has_value()) << "output_embeds output not found on prefill model";
    ASSERT_TRUE(embeds->get_partial_shape().is_static());
    EXPECT_EQ(embeds->get_shape(), (ov::Shape{1, 1, 64}));
    EXPECT_TRUE(output_embeds_has_gather_producer(prefill.model));
    EXPECT_TRUE(find_input(prefill.model, ov::npuw::util::kLogitsSelectionIndicesParamName).has_value());
    EXPECT_EQ(lm_head.model->input(0).get_shape(), (ov::Shape{1, 1, 64}));
}

// Test 3: When SHARED_HEAD=NO, SliceOutEmbeds does not fire. The lm_head sub-model
// must not be created and the prefill model must not expose an output_embeds output.
TEST_F(SliceOutEmbedsPassTest, SliceNotAppliedWhenNoSharedHead) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(
        compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"}, {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
                                         recorder));
    ASSERT_NE(compiled, nullptr);

    // lm_head sub-model must not exist when SHARED_HEAD=NO
    const auto* lm_head = recorder.find_suffix("_lm_head");
    EXPECT_EQ(lm_head, nullptr) << "lm_head sub-model should not exist when SHARED_HEAD=NO";

    // prefill model must not export output_embeds
    const auto& prefill = require_sub_model(recorder, "_prefill");

    const auto embeds = find_output(prefill.model, ov::npuw::LLMCompiledModel::layer_names::output_embeds);
    EXPECT_FALSE(embeds.has_value())
        << "output_embeds output should not exist in the prefill model when SHARED_HEAD=NO";
}

}  // namespace
