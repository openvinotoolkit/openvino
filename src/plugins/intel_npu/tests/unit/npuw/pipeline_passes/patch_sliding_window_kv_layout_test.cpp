// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for PatchSlidingWindowKVLayout through the LLMCompiledModel pipeline.
// Focus: sliding-mask externalization and sliding/full layer shape invariants.

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "kv_cache_sliding_window_manager.hpp"
#include "llm_pass_test_fixture.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class PatchSlidingWindowKVLayoutTest : public ov::test::npuw::LLMPassTestFixture {};

// Maps layer index parsed from "model.layers.N.self_attn" to SDPA node.
std::map<size_t, std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> sdpa_by_layer(
    const std::shared_ptr<ov::Model>& model) {
    std::map<size_t, std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> result;
    const std::string marker = "model.layers.";
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }
        const auto& name = sdpa->get_friendly_name();
        auto pos = name.find(marker);
        if (pos == std::string::npos) {
            continue;
        }
        result[std::stoul(name.substr(pos + marker.size()))] = sdpa;
    }
    return result;
}

constexpr size_t kWindowSize = 32;  // deliberately << default kvcache_size (192) to be distinguishable.
constexpr size_t kNumLayers = 4;    // layers 0,2 sliding; layers 1,3 full-attention.

std::shared_ptr<ov::Model> build_hybrid_model() {
    return ov::test::npuw::build_sliding_window_test_model(kWindowSize, /*sliding_to_full_ratio=*/1, {}, kNumLayers);
}

}  // namespace

// Hybrid model invariant: both generate and prefill expose only the sliding-mask input.
TEST_F(PatchSlidingWindowKVLayoutTest, PrefillAndGenerate_ExternalizeOnlySlidingMask) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");
    const auto& prefill = require_sub_model(recorder, "_prefill");

    EXPECT_EQ(count_inputs(generate.model, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(generate.model, "global_attention_mask"), 0u);
    EXPECT_EQ(count_inputs(prefill.model, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(prefill.model, "global_attention_mask"), 0u);
}

// Baseline invariant: non-hybrid models must not gain sliding_window_attention_mask.
TEST_F(PatchSlidingWindowKVLayoutTest, NonHybridModel_NoMaskExternalized) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));  // default build_llm_test_model(), no SWA.
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_EQ(count_inputs(generate.model, "sliding_window_attention_mask"), 0u);
    EXPECT_EQ(count_inputs(generate.model, "global_attention_mask"), 0u);
}

// Layer-selective invariant: only sliding layers shrink the past-KV sequence axis.
TEST_F(PatchSlidingWindowKVLayoutTest, GenerateModel_OnlySlidingLayersPastKVShrunk) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // Default props: kvcache_size=192, generate input_size=1 => full past=191.
    const auto sliding_past = input_shape(generate.model, "past_key_values.0.key");
    ASSERT_TRUE(sliding_past.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_EQ((*sliding_past)[2], kWindowSize);

    const auto full_past = input_shape(generate.model, "past_key_values.1.key");
    ASSERT_TRUE(full_past.has_value()) << "past_key_values.1.key not found in generate model";
    EXPECT_EQ((*full_past)[2], 191u);
}

// Generate invariant: sliding mask width matches the post-concat KV total width.
TEST_F(PatchSlidingWindowKVLayoutTest, GenerateModel_SlidingMaskWidthShrunkToNewKvTotal) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");
    const auto mask_shape = input_shape(generate.model, "sliding_window_attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "sliding_window_attention_mask not found in generate model";

    // Generate input_size=1, window=32 => new_kv_total=33.
    EXPECT_EQ(mask_shape->back(), 33u);
}

// Prefill invariant: sliding mask width follows the same post-concat KV width rule.
TEST_F(PatchSlidingWindowKVLayoutTest, PrefillModel_SlidingMaskWidthShrunkToNewKvTotal) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto mask_shape = input_shape(prefill.model, "sliding_window_attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "sliding_window_attention_mask not found in prefill model";
    const auto input_ids_shape = input_shape(prefill.model, "input_ids");
    ASSERT_TRUE(input_ids_shape.has_value()) << "input_ids not found in prefill model";
    const auto sliding_past = input_shape(prefill.model, "past_key_values.0.key");

    const size_t expected_width =
        sliding_past.has_value() ? ((*sliding_past)[2] + input_ids_shape->back()) : input_ids_shape->back();

    // Invariant: mask width = post-concat total width.
    EXPECT_EQ(mask_shape->back(), expected_width);
}

// Only sliding-layer past KV parameters get SWA rt_info tag.
TEST_F(PatchSlidingWindowKVLayoutTest, GenerateModel_OnlySlidingLayersPastKVTaggedWithRtInfo) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    const auto sliding_input = find_input(generate.model, "past_key_values.0.key");
    ASSERT_TRUE(sliding_input.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_TRUE(sliding_input->get_node()->get_rt_info().count(ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY) > 0);

    const auto full_input = find_input(generate.model, "past_key_values.1.key");
    ASSERT_TRUE(full_input.has_value()) << "past_key_values.1.key not found in generate model";
    EXPECT_EQ(full_input->get_node()->get_rt_info().count(ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY), 0u);
}

// Input-source invariant: sliding SDPA consumes the externalized mask, full-attention SDPA does not.
TEST_F(PatchSlidingWindowKVLayoutTest, GenerateModel_SlidingSDPAUsesExternalizedMask_FullSDPADoesNot) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // Keep SDPA nodes intact for direct mask-input inspection.
    ASSERT_NO_THROW(compiled =
                        create_compiled_model(build_hybrid_model(), {{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    const auto sdpa_map = sdpa_by_layer(generate.model);
    ASSERT_EQ(sdpa_map.size(), kNumLayers);

    auto is_sliding_mask_param = [](const ov::Node* node) {
        auto param = ov::as_type<const ov::op::v0::Parameter>(node);
        return param != nullptr && param->get_friendly_name() == "sliding_window_attention_mask";
    };

    // Sliding layers (0, 2): mask input IS the externalized Parameter.
    EXPECT_TRUE(is_sliding_mask_param(sdpa_map.at(0)->input_value(3).get_node()));
    EXPECT_TRUE(is_sliding_mask_param(sdpa_map.at(2)->input_value(3).get_node()));

    // Full-attention layers (1, 3): mask input is NOT the externalized Parameter.
    EXPECT_FALSE(is_sliding_mask_param(sdpa_map.at(1)->input_value(3).get_node()));
    EXPECT_FALSE(is_sliding_mask_param(sdpa_map.at(3)->input_value(3).get_node()));
}

// With default prefill hint and chunk size equal to max prompt length,
// prefill drops empty past KV inputs while generate keeps window-sized past; both externalize SWA mask.
TEST_F(PatchSlidingWindowKVLayoutTest, PrefillAndGenerate_ExpectedPastAndMaskBehaviorForPromptAndTotalKv) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(),
                                                     {{"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                                      {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                                      {"NPUW_LLM_PREFILL_CHUNK_SIZE", "128"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_EQ(count_inputs(prefill.model, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(generate.model, "sliding_window_attention_mask"), 1u);

    const auto input_ids_shape = input_shape(prefill.model, "input_ids");
    ASSERT_TRUE(input_ids_shape.has_value()) << "input_ids not found in prefill model";

    const auto prefill_mask_shape = input_shape(prefill.model, "sliding_window_attention_mask");
    ASSERT_TRUE(prefill_mask_shape.has_value()) << "sliding_window_attention_mask not found in prefill model";
    const auto generate_mask_shape = input_shape(generate.model, "sliding_window_attention_mask");
    ASSERT_TRUE(generate_mask_shape.has_value()) << "sliding_window_attention_mask not found in generate model";

    EXPECT_EQ(count_inputs(prefill.model, "past_key_values.0.key"), 0u);

    const auto generate_sliding_past = input_shape(generate.model, "past_key_values.0.key");
    ASSERT_TRUE(generate_sliding_past.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_EQ((*generate_sliding_past)[2], kWindowSize);

    EXPECT_EQ(prefill_mask_shape->back(), input_ids_shape->back());
    EXPECT_EQ(prefill_mask_shape->back(), 128u);
    EXPECT_EQ(generate_mask_shape->back(), 33u);
}
