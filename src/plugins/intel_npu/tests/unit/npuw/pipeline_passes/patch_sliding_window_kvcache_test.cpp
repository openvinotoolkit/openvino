// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for ov::npuw::PatchSlidingWindowKVCache, exercised through the full
// LLMCompiledModel pipeline (DetectAttentionMask -> ReshapeToStatic ->
// PatchSlidingWindowKVCache), the same path production code uses. See
// npuw_transformations/patch_sliding_window_kvcache.hpp for the pass' design.
//
// Coverage focus: for a genuine hybrid SWA model (some layers sliding, some
// full-attention), only `sliding_window_attention_mask` is externalized as a
// new model input; `global_attention_mask` must never be created, and
// full-attention SDPAs must keep their original (non-externalized) mask input.

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class PatchSlidingWindowKVCacheTest : public ov::test::npuw::LLMPassTestFixture {};

// Maps layer index (parsed from "model.layers.N.self_attn" in the SDPA's own
// friendly name) to its SDPA node, for a 4-layer hybrid model built with
// sliding_to_full_ratio=1 (layers 0,2 sliding; layers 1,3 full-attention -
// see build_sliding_window_test_model()/model_builder.cpp's cycle logic).
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

constexpr size_t kWindowSize = 32;   // deliberately << default kvcache_size (192) to be distinguishable.
constexpr size_t kNumLayers = 4;     // layers 0,2 sliding; layers 1,3 full-attention.

std::shared_ptr<ov::Model> build_hybrid_model() {
    return ov::test::npuw::build_sliding_window_test_model(kWindowSize, /*sliding_to_full_ratio=*/1, {}, kNumLayers);
}

}  // namespace

// Generate sub-model: exactly one `sliding_window_attention_mask` input, and
// no `global_attention_mask` input at all.
TEST_F(PatchSlidingWindowKVCacheTest, GenerateModel_ExternalizesOnlySlidingMask) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    auto hybrid = build_hybrid_model();
    ASSERT_NO_THROW(compiled = create_compiled_model(hybrid, {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_EQ(count_inputs(generate.model, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(generate.model, "global_attention_mask"), 0u);
}

// Prefill sub-model: same expectation - only the sliding mask is externalized.
TEST_F(PatchSlidingWindowKVCacheTest, PrefillModel_ExternalizesOnlySlidingMask) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");

    EXPECT_EQ(count_inputs(prefill.model, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(prefill.model, "global_attention_mask"), 0u);
}

// A non-hybrid model (no sliding layers at all) must never gain a
// sliding_window_attention_mask input - the pass is a no-op there.
TEST_F(PatchSlidingWindowKVCacheTest, NonHybridModel_NoMaskExternalized) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({}, recorder));  // default build_llm_test_model(), no SWA.
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_EQ(count_inputs(generate.model, "sliding_window_attention_mask"), 0u);
    EXPECT_EQ(count_inputs(generate.model, "global_attention_mask"), 0u);
}

// Step 1 must shrink past_key_values only for sliding layers (0, 2): their past
// axis becomes exactly window_size. Full-attention layers (1, 3) keep the
// standard (kvcache_size - input_size) sizing, untouched by this pass.
TEST_F(PatchSlidingWindowKVCacheTest, GenerateModel_OnlySlidingLayersPastKVShrunk) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(build_hybrid_model(), {}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // Default props: NPUW_LLM_MAX_PROMPT_LEN=128, NPUW_LLM_MIN_RESPONSE_LEN=64 -> kvcache_size=192.
    // Generate variant: input_size=1 -> full-attention past = 192 - 1 = 191.
    const auto sliding_past = input_shape(generate.model, "past_key_values.0.key");
    ASSERT_TRUE(sliding_past.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_EQ((*sliding_past)[2], kWindowSize);

    const auto full_past = input_shape(generate.model, "past_key_values.1.key");
    ASSERT_TRUE(full_past.has_value()) << "past_key_values.1.key not found in generate model";
    EXPECT_EQ((*full_past)[2], 191u);
}

// The sliding SDPA's mask input must be the externalized
// sliding_window_attention_mask Parameter. The full-attention SDPA's mask
// input must NOT be that Parameter - its own original mask representation is
// left completely untouched by this pass.
TEST_F(PatchSlidingWindowKVCacheTest, GenerateModel_SlidingSDPAUsesExternalizedMask_FullSDPADoesNot) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    // Disable the (default-on) V-tensor/SDPA-decomposition optimization so the SDPA nodes
    // this test inspects survive intact into the recorded compiled model - that optimization
    // is unrelated to what PatchSlidingWindowKVCache itself does and runs strictly after it.
    ASSERT_NO_THROW(
        compiled = create_compiled_model(build_hybrid_model(), {{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}}, recorder));
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
