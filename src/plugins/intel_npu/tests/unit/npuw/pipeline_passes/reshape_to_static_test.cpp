// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class ReshapeToStaticPassTest : public ov::test::npuw::LLMPassTestFixture {};

// --- Test 1 -------------------------------------------------------------------------
// Every input of the prefill sub-model must be fully static after ReshapeToStatic.
// We additionally verify the concrete shapes of the two most representative inputs.
TEST_F(ReshapeToStaticPassTest, AllPrefillInputsAreStatic) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // Default props: MAX_PROMPT_LEN=128, MIN_RESPONSE_LEN=64
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");

    EXPECT_TRUE(all_inputs_static(prefill.model))
        << "At least one prefill input still has a dynamic dimension after ReshapeToStatic";

    // Spot-check concrete shapes
    const auto ids_shape = input_shape(prefill.model, "input_ids");
    ASSERT_TRUE(ids_shape.has_value()) << "input_ids not found in prefill model";
    EXPECT_EQ(*ids_shape, (ov::Shape{1, 128}));

    const auto mask_shape = input_shape(prefill.model, "attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "attention_mask not found in prefill model";
    EXPECT_EQ(*mask_shape, (ov::Shape{1, 128}));
}

// --- Test 2 -------------------------------------------------------------------------
// Every input of the generate sub-model must be fully static after ReshapeToStatic.
// kvcache_size = 128 + 64 = 192, so the model is named _kv192.
// Model config: num_kv_heads=4, head_dim=16, max_generation_token_len=1 (default).
TEST_F(ReshapeToStaticPassTest, AllGenerateInputsAreStatic) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // Default props: MAX_PROMPT_LEN=128, MIN_RESPONSE_LEN=64, single-token generate
    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(all_inputs_static(generate.model))
        << "At least one generate input still has a dynamic dimension after ReshapeToStatic";

    // input_ids: one token at a time (default max_generation_token_len=1)
    const auto ids_shape = input_shape(generate.model, "input_ids");
    ASSERT_TRUE(ids_shape.has_value()) << "input_ids not found in generate model";
    EXPECT_EQ(*ids_shape, (ov::Shape{1, 1}));

    // attention_mask covers the full kvcache window
    const auto mask_shape = input_shape(generate.model, "attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "attention_mask not found in generate model";
    EXPECT_EQ(*mask_shape, (ov::Shape{1, 192}));

    // past_key_values: [batch=1, num_kv_heads=4, past_seq=kvcache-input_size=191, head_dim=16]
    const auto kv_shape = input_shape(generate.model, "past_key_values");
    ASSERT_TRUE(kv_shape.has_value()) << "past_key_values not found in generate model";
    ASSERT_EQ(kv_shape->size(), 4u);
    EXPECT_EQ((*kv_shape)[0], 1u);   // batch
    EXPECT_EQ((*kv_shape)[1], 4u);   // num_kv_heads
    EXPECT_EQ((*kv_shape)[2], 191u); // kvcache_size(192) - input_size(1)
    EXPECT_EQ((*kv_shape)[3], 16u);  // head_dim
}

// --- Test 3 -------------------------------------------------------------------------
// The past_key_values sequence dimension in the generate model must equal
// kvcache_size - input_size.  With MAX_PROMPT_LEN=256 and MIN_RESPONSE_LEN=128:
//   kvcache_size = 256 + 128 = 384
//   past seq dim = 384 - 1 = 383
TEST_F(ReshapeToStaticPassTest, GenerateModelKVCacheShapeReflectsKVCacheSize) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_MAX_PROMPT_LEN", "256"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    // kvcache_size = 256 + 128 = 384
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(all_inputs_static(generate.model))
        << "At least one generate input still has a dynamic dimension after ReshapeToStatic";

    const auto kv_shape = input_shape(generate.model, "past_key_values");
    ASSERT_TRUE(kv_shape.has_value()) << "past_key_values not found in generate model";
    ASSERT_EQ(kv_shape->size(), 4u);
    // seq dim = kvcache_size(384) - input_size(1) = 383
    EXPECT_EQ((*kv_shape)[2], 383u);

    // Sanity-check the other dims while we're here
    EXPECT_EQ((*kv_shape)[0], 1u);   // batch
    EXPECT_EQ((*kv_shape)[1], 4u);   // num_kv_heads
    EXPECT_EQ((*kv_shape)[3], 16u);  // head_dim
}

// --- Test 4 -------------------------------------------------------------------------
// When NPUW_LLM_MAX_GENERATION_TOKEN_LEN=8:
//   * input_ids in the generate model has shape {1, 8}
//   * past_key_values seq dim = kvcache_size(192) - input_size(8) = 184
TEST_F(ReshapeToStaticPassTest, MaxGenerationTokenLenDrivesGenerateInputShape) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_MAX_GENERATION_TOKEN_LEN", "8"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    // MAX_PROMPT_LEN=128, MIN_RESPONSE_LEN=64 from base_props -> kvcache_size=192
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(all_inputs_static(generate.model))
        << "At least one generate input still has a dynamic dimension after ReshapeToStatic";

    // input_ids must accommodate MAX_GENERATION_TOKEN_LEN tokens
    const auto ids_shape = input_shape(generate.model, "input_ids");
    ASSERT_TRUE(ids_shape.has_value()) << "input_ids not found in generate model";
    EXPECT_EQ(*ids_shape, (ov::Shape{1, 8}));

    // attention_mask still spans the full kvcache window
    const auto mask_shape = input_shape(generate.model, "attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "attention_mask not found in generate model";
    EXPECT_EQ(*mask_shape, (ov::Shape{1, 192}));

    // past_key_values seq dim = 192 - 8 = 184
    const auto kv_shape = input_shape(generate.model, "past_key_values");
    ASSERT_TRUE(kv_shape.has_value()) << "past_key_values not found in generate model";
    ASSERT_EQ(kv_shape->size(), 4u);
    EXPECT_EQ((*kv_shape)[0], 1u);   // batch
    EXPECT_EQ((*kv_shape)[1], 4u);   // num_kv_heads
    EXPECT_EQ((*kv_shape)[2], 184u); // kvcache_size(192) - input_size(8)
    EXPECT_EQ((*kv_shape)[3], 16u);  // head_dim
}

}  // namespace


// ---------------------------------------------------------------------------
// Direct ReshapeToStatic unit tests for Qwen3-VL inputs
// ---------------------------------------------------------------------------
// These tests build a minimal model graph with only the VLM-specific inputs
// and apply ReshapeToStatic directly, without going through LLMCompiledModel.
// This keeps the tests fast and free of plugin/mock overhead.
// ---------------------------------------------------------------------------

namespace {

using ov::npuw::KVAxesPosition;
using ov::npuw::ReshapeToStatic;

// Build a minimal model that contains visual_pos_masks and attention_mask.
// Both are dynamic so ReshapeToStatic must assign static shapes to them.
static std::shared_ptr<ov::Model> build_vlm_inputs_model() {
    auto visual_pos_masks =
        std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    visual_pos_masks->set_friendly_name("visual_pos_masks");
    visual_pos_masks->output(0).set_names({"visual_pos_masks"});

    auto attention_mask =
        std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{-1, -1});
    attention_mask->set_friendly_name("attention_mask");
    attention_mask->output(0).set_names({"attention_mask"});

    auto r1 = std::make_shared<ov::op::v0::Result>(visual_pos_masks);
    auto r2 = std::make_shared<ov::op::v0::Result>(attention_mask);

    return std::make_shared<ov::Model>(
        ov::ResultVector{r1, r2},
        ov::ParameterVector{visual_pos_masks, attention_mask},
        "vlm_inputs_model");
}

// --- Test 5 -------------------------------------------------------------------
// visual_pos_masks must be reshaped to {2, input_size}.
// The Whisper kvcache override (lhs_seq_size && kvcache_size > 4) must NOT
// affect visual_pos_masks — even when lhs_seq_size is non-zero.
TEST(ReshapeToStaticVLMTest, VisualPosMasksShapeIsRowsByInputSize) {
    auto model = build_vlm_inputs_model();

    constexpr uint32_t input_size = 128;
    constexpr uint32_t kvcache_size = 192;
    KVAxesPosition kv_axes{0, 1};

    // lhs_seq_size=8 would trigger the Whisper branch for attention_mask;
    // visual_pos_masks must NOT be overwritten by it.
    ReshapeToStatic pass(input_size, kvcache_size, kv_axes, /*lora_rank=*/0, /*lhs_seq_size=*/8);
    ASSERT_TRUE(pass.run_on_model(model));

    bool found = false;
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        if (names.count("visual_pos_masks")) {
            found = true;
            const auto& shape = input.get_partial_shape();
            ASSERT_TRUE(shape.is_static()) << "visual_pos_masks must be static after ReshapeToStatic";
            EXPECT_EQ(shape.to_shape(), (ov::Shape{2, input_size}))
                << "visual_pos_masks must have shape {2, input_size}";
        }
    }
    EXPECT_TRUE(found) << "visual_pos_masks input not found in model";
}

// --- Test 6 -------------------------------------------------------------------
// Sanity check: attention_mask IS affected by lhs_seq_size (Whisper path).
// This test documents the intentional difference between visual_pos_masks
// and attention_mask under the same Whisper-mode conditions.
TEST(ReshapeToStaticVLMTest, AttentionMaskIsAffectedByWhisperPath) {
    auto model = build_vlm_inputs_model();

    constexpr uint32_t input_size = 128;
    constexpr uint32_t kvcache_size = 192;
    KVAxesPosition kv_axes{0, 1};

    ReshapeToStatic pass(input_size, kvcache_size, kv_axes, /*lora_rank=*/0, /*lhs_seq_size=*/8);
    ASSERT_TRUE(pass.run_on_model(model));

    bool found = false;
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        if (names.count("attention_mask")) {
            found = true;
            const auto& shape = input.get_partial_shape();
            ASSERT_TRUE(shape.is_static());
            // Whisper path: kvcache_size + 1
            EXPECT_EQ(shape.to_shape(), (ov::Shape{1, kvcache_size + 1}));
        }
    }
    EXPECT_TRUE(found) << "attention_mask input not found in model";
}

}  // namespace
