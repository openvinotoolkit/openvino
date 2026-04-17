// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class SliceOutEmbedsPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    // Returns true iff the direct producer of the output_embeds Result is a v8::Slice.
    // This is precisely what SliceOutEmbeds inserts; other Slice ops (e.g. from RoPE
    // half-rotation) are NOT on this path and must not be counted.
    static bool output_embeds_has_slice_producer(const std::shared_ptr<ov::Model>& model) {
        for (const auto& output : model->outputs()) {
            for (const auto& name : output.get_names()) {
                if (name.find(ov::npuw::LLMCompiledModel::layer_names::output_embeds) != std::string::npos) {
                    // output is an Output<const Node> over the Result node
                    auto result_node = output.get_node_shared_ptr();
                    if (!result_node || result_node->inputs().empty())
                        return false;
                    auto producer = result_node->input(0).get_source_output().get_node_shared_ptr();
                    return ov::is_type<ov::op::v8::Slice>(producer);
                }
            }
        }
        return false;
    }
};

// Test 1: When SHARED_HEAD=YES and MAX_GENERATION_TOKEN_LEN < MAX_PROMPT_LEN,
// SliceOutEmbeds should insert a v8::Slice on the output_embeds path and the
// resulting output shape should be {1, MAX_GENERATION_TOKEN_LEN, hidden_size}.
TEST_F(SliceOutEmbedsPassTest, SliceIsInsertedWhenGenerationTokenLenIsLessThanPromptLen) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
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

    // The direct producer of the output_embeds Result must be a v8::Slice (inserted by SliceOutEmbeds).
    // Other v8::Slice ops (e.g. from HalfRotationRoPE halving) are intentionally not counted.
    EXPECT_TRUE(output_embeds_has_slice_producer(prefill.model))
        << "Expected a v8::Slice directly feeding output_embeds, but none was found";
}

// Test 2: When SHARED_HEAD=YES but MAX_GENERATION_TOKEN_LEN == MAX_PROMPT_LEN,
// SliceOutEmbeds is a no-op: no Slice is inserted and the shape remains full.
TEST_F(SliceOutEmbedsPassTest, SliceIsNotInsertedWhenGenerationTokenLenEqualsPromptLen) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // MAX_GENERATION_TOKEN_LEN=128 equals MAX_PROMPT_LEN=128, so no slicing is needed.
    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "YES"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "128"}},
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

    // The direct producer of the output_embeds Result must NOT be a v8::Slice.
    // (Other v8::Slices from RoPE etc. are irrelevant and deliberately ignored.)
    EXPECT_FALSE(output_embeds_has_slice_producer(prefill.model))
        << "Expected no v8::Slice directly feeding output_embeds when generation len == prompt len";
}

// Test 3: When SHARED_HEAD=NO, SliceOutEmbeds does not fire. The lm_head sub-model
// must not be created and the prefill model must not expose an output_embeds output.
TEST_F(SliceOutEmbedsPassTest, SliceNotAppliedWhenNoSharedHead) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_SHARED_HEAD", "NO"},
                                                      {"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}},
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
