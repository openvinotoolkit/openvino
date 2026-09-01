// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Tests for ShrinkSlidingWindowKVCache.
// Focus: sliding-mask externalization and sliding/full layer shape invariants.
//
// The pass is invoked directly rather than through LLMCompiledModel, so the tests
// stay independent of pipeline wiring. run_shrink_pass() reproduces exactly the
// preparation LLMCompiledModel performs right before this pass: annotate per-SDPA
// mask kinds, derive the SWA layout, then reshape to static.

#include "npuw_transformations/shrink_sliding_window_kv_cache.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "kv_cache_sliding_window_manager.hpp"
#include "llm_pass_test_fixture.hpp"
#include "npuw_transformations/add_position_ids_param.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "npuw_transformations/kv_axes_position.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {

class ShrinkSlidingWindowKVCacheTest : public ov::test::npuw::LLMPassTestFixture {};

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

constexpr size_t kWindowSize = 32;  // deliberately << kvcache_size (192) to be distinguishable.
constexpr size_t kNumLayers = 4;    // layers 0,2 sliding; layers 1,3 full-attention.

// Mirror of the default LLMPassTestFixture properties: max prompt 128, min response 64.
constexpr uint32_t kMaxPromptLen = 128;
constexpr uint32_t kKvCacheSize = 192;
constexpr uint32_t kGenerateInputSize = 1;
const ov::npuw::KVAxesPosition kAxes{0u, 2u};

std::shared_ptr<ov::Model> build_hybrid_model() {
    return ov::test::npuw::build_sliding_window_test_model(kWindowSize, /*sliding_to_full_ratio=*/1, {}, kNumLayers);
}

// GQA variant: num_kv_heads < num_heads makes repeat_kv insert a Broadcast whose target
// shape carries the KV length, which is exactly what the pass has to privatize and patch.
std::shared_ptr<ov::Model> build_hybrid_gqa_model() {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.num_layers = kNumLayers;
    cfg.num_kv_heads = 2;
    cfg.sliding_window_size = kWindowSize;
    cfg.sliding_to_full_ratio = 1;
    ov::test::npuw::ModelBuilder mb;
    return mb.build_llm(cfg);
}

// Runs the same preparation sequence LLMCompiledModel applies before the pass under test.
std::shared_ptr<ov::Model> run_shrink_pass(const std::shared_ptr<ov::Model>& model,
                                           uint32_t input_size,
                                           uint32_t kvcache_size,
                                           bool is_prefill) {
    ov::npuw::AddPositionIdsParam().run_on_model(model);
    ov::pass::StatefulToStateless().run_on_model(model);
    ov::npuw::DetectAttentionMask().run_on_model(model);
    const auto layout = ov::npuw::util::detect_swa_layout(model);
    ov::npuw::ReshapeToStatic(input_size, kvcache_size, kAxes, /*lora_rank=*/0, /*lhs_seq_size=*/0, is_prefill)
        .run_on_model(model);
    ov::npuw::ShrinkSlidingWindowKVCache(layout, kvcache_size, input_size, kAxes).run_on_model(model);
    return model;
}

std::shared_ptr<ov::Model> make_generate_model(const std::shared_ptr<ov::Model>& model) {
    return run_shrink_pass(model, kGenerateInputSize, kKvCacheSize, /*is_prefill=*/false);
}

std::shared_ptr<ov::Model> make_prefill_model(const std::shared_ptr<ov::Model>& model) {
    return run_shrink_pass(model, kMaxPromptLen, kMaxPromptLen, /*is_prefill=*/true);
}

}  // namespace

// Hybrid model invariant: both generate and prefill expose only the sliding-mask input.
TEST_F(ShrinkSlidingWindowKVCacheTest, PrefillAndGenerate_ExternalizeOnlySlidingMask) {
    std::shared_ptr<ov::Model> generate;
    std::shared_ptr<ov::Model> prefill;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));
    ASSERT_NO_THROW(prefill = make_prefill_model(build_hybrid_model()));

    EXPECT_EQ(count_inputs(generate, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(generate, "global_attention_mask"), 0u);
    EXPECT_EQ(count_inputs(prefill, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(prefill, "global_attention_mask"), 0u);
}

// Baseline invariant: non-hybrid models must not gain sliding_window_attention_mask.
TEST_F(ShrinkSlidingWindowKVCacheTest, NonHybridModel_NoMaskExternalized) {
    std::shared_ptr<ov::Model> generate;
    // Default build_llm_test_model(), no SWA.
    ASSERT_NO_THROW(generate = make_generate_model(ov::test::npuw::build_llm_test_model()));

    EXPECT_EQ(count_inputs(generate, "sliding_window_attention_mask"), 0u);
    EXPECT_EQ(count_inputs(generate, "global_attention_mask"), 0u);
}

// Layer-selective invariant: only sliding layers shrink the past-KV sequence axis.
TEST_F(ShrinkSlidingWindowKVCacheTest, GenerateModel_OnlySlidingLayersPastKVShrunk) {
    std::shared_ptr<ov::Model> generate;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));

    // kvcache_size=192, generate input_size=1 => full past=191.
    const auto sliding_past = input_shape(generate, "past_key_values.0.key");
    ASSERT_TRUE(sliding_past.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_EQ((*sliding_past)[2], kWindowSize);

    const auto full_past = input_shape(generate, "past_key_values.1.key");
    ASSERT_TRUE(full_past.has_value()) << "past_key_values.1.key not found in generate model";
    EXPECT_EQ((*full_past)[2], 191u);
}

// Generate invariant: sliding mask width matches the post-concat KV total width.
TEST_F(ShrinkSlidingWindowKVCacheTest, GenerateModel_SlidingMaskWidthShrunkToNewKvTotal) {
    std::shared_ptr<ov::Model> generate;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));

    const auto mask_shape = input_shape(generate, "sliding_window_attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "sliding_window_attention_mask not found in generate model";

    // Generate input_size=1, window=32 => new_kv_total=33.
    EXPECT_EQ(mask_shape->back(), 33u);
}

// Prefill invariant: sliding mask width follows the same post-concat KV width rule.
TEST_F(ShrinkSlidingWindowKVCacheTest, PrefillModel_SlidingMaskWidthShrunkToNewKvTotal) {
    std::shared_ptr<ov::Model> prefill;
    ASSERT_NO_THROW(prefill = make_prefill_model(build_hybrid_model()));

    const auto mask_shape = input_shape(prefill, "sliding_window_attention_mask");
    ASSERT_TRUE(mask_shape.has_value()) << "sliding_window_attention_mask not found in prefill model";
    const auto input_ids_shape = input_shape(prefill, "input_ids");
    ASSERT_TRUE(input_ids_shape.has_value()) << "input_ids not found in prefill model";
    const auto sliding_past = input_shape(prefill, "past_key_values.0.key");

    const size_t expected_width =
        sliding_past.has_value() ? ((*sliding_past)[2] + input_ids_shape->back()) : input_ids_shape->back();

    // Invariant: mask width = post-concat total width.
    EXPECT_EQ(mask_shape->back(), expected_width);
}

// Only sliding-layer past KV parameters get SWA rt_info tag.
TEST_F(ShrinkSlidingWindowKVCacheTest, GenerateModel_OnlySlidingLayersPastKVTaggedWithRtInfo) {
    std::shared_ptr<ov::Model> generate;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));

    const auto sliding_input = find_input(generate, "past_key_values.0.key");
    ASSERT_TRUE(sliding_input.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_TRUE(sliding_input->get_node()->get_rt_info().count(ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY) > 0);

    const auto full_input = find_input(generate, "past_key_values.1.key");
    ASSERT_TRUE(full_input.has_value()) << "past_key_values.1.key not found in generate model";
    EXPECT_EQ(full_input->get_node()->get_rt_info().count(ov::npuw::util::NPUW_KV_CACHE_SLIDING_RT_KEY), 0u);
}

// Input-source invariant: sliding SDPA consumes the externalized mask, full-attention SDPA does not.
TEST_F(ShrinkSlidingWindowKVCacheTest, GenerateModel_SlidingSDPAUsesExternalizedMask_FullSDPADoesNot) {
    std::shared_ptr<ov::Model> generate;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));

    const auto sdpa_map = sdpa_by_layer(generate);
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

// Single-shot prefill (input_size == kvcache_size) leaves sliding layers with no past region,
// while generate keeps a window-sized past. Both externalize the SWA mask.
TEST_F(ShrinkSlidingWindowKVCacheTest, PrefillAndGenerate_ExpectedPastAndMaskBehaviorForPromptAndTotalKv) {
    std::shared_ptr<ov::Model> generate;
    std::shared_ptr<ov::Model> prefill;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_model()));
    ASSERT_NO_THROW(prefill = make_prefill_model(build_hybrid_model()));

    EXPECT_EQ(count_inputs(prefill, "sliding_window_attention_mask"), 1u);
    EXPECT_EQ(count_inputs(generate, "sliding_window_attention_mask"), 1u);

    const auto input_ids_shape = input_shape(prefill, "input_ids");
    ASSERT_TRUE(input_ids_shape.has_value()) << "input_ids not found in prefill model";

    const auto prefill_mask_shape = input_shape(prefill, "sliding_window_attention_mask");
    ASSERT_TRUE(prefill_mask_shape.has_value()) << "sliding_window_attention_mask not found in prefill model";
    const auto generate_mask_shape = input_shape(generate, "sliding_window_attention_mask");
    ASSERT_TRUE(generate_mask_shape.has_value()) << "sliding_window_attention_mask not found in generate model";

    // available_past == 0 => sliding layers keep no past KV at all.
    const auto prefill_sliding_past = input_shape(prefill, "past_key_values.0.key");
    ASSERT_TRUE(prefill_sliding_past.has_value()) << "past_key_values.0.key not found in prefill model";
    EXPECT_EQ((*prefill_sliding_past)[2], 0u);

    const auto generate_sliding_past = input_shape(generate, "past_key_values.0.key");
    ASSERT_TRUE(generate_sliding_past.has_value()) << "past_key_values.0.key not found in generate model";
    EXPECT_EQ((*generate_sliding_past)[2], kWindowSize);

    EXPECT_EQ(prefill_mask_shape->back(), input_ids_shape->back());
    EXPECT_EQ(prefill_mask_shape->back(), 128u);
    EXPECT_EQ(generate_mask_shape->back(), 33u);
}

// Shape-privatization invariant: KV target-shape constants that carried the full kvcache
// length are replaced by private constants holding the new post-concat KV total.
TEST_F(ShrinkSlidingWindowKVCacheTest, GenerateModel_SlidingKVShapeConstantsPatchedToNewKvTotal) {
    std::shared_ptr<ov::Model> generate;
    ASSERT_NO_THROW(generate = make_generate_model(build_hybrid_gqa_model()));

    // Generate input_size=1, window=32 => new_kv_total=33.
    constexpr int64_t kNewKvTotal = 33;
    std::size_t num_patched = 0;
    for (const auto& op : generate->get_ordered_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || constant->get_friendly_name().find("/swa_kv_patched") == std::string::npos) {
            continue;
        }
        const auto vals = constant->cast_vector<int64_t>();
        ASSERT_GE(vals.size(), 2u) << constant->get_friendly_name();
        EXPECT_EQ(vals[vals.size() - 2], kNewKvTotal) << constant->get_friendly_name();
        ++num_patched;
    }

    // Sliding layers (0, 2) only, each patching its K and V repeat_kv Broadcast.
    EXPECT_EQ(num_patched, 4u);
}
