// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_flash_attention.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "common_test_utils/node_builders/constant.hpp"
#include "host_mask_catalogue_models.hpp"
#include "model_builder_masks.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "npuw_transformations/propagate_slice.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "serialization.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

namespace {

static constexpr size_t BATCH = 1;
static constexpr size_t NUM_HEADS = 8;
static constexpr size_t HEAD_DIM = 64;
static constexpr size_t QUERY_SIZE = 16;
static constexpr size_t PAST_LEN = 48;

// Build a minimal SDPA model that HostFlashAttention::from() can parse.
std::shared_ptr<ov::Model> build_sdpa_model(size_t query_size = QUERY_SIZE,
                                            size_t past_len = PAST_LEN,
                                            size_t num_heads = NUM_HEADS,
                                            size_t head_dim = HEAD_DIM) {
    using namespace ov;

    const size_t context_size = past_len + query_size;
    const Shape past_shape = {BATCH, num_heads, past_len, head_dim};
    const Shape new_shape = {BATCH, num_heads, query_size, head_dim};
    const Shape mask_shape = {BATCH, 1, query_size, context_size};

    ResultVector results;

    auto query = ov::test::utils::make_param(element::f32, new_shape, "query.0");
    auto past_key = ov::test::utils::make_param(element::f32, past_shape, "past_key_values.0.key");
    auto past_val = ov::test::utils::make_param(element::f32, past_shape, "past_key_values.0.value");
    auto new_key = ov::test::utils::make_param(element::f32, new_shape, "new_key.0");
    auto new_val = ov::test::utils::make_param(element::f32, new_shape, "new_value.0");
    auto mask = ov::test::utils::make_param(element::f32, mask_shape, "mask.0");
    ParameterVector params = {query, past_key, past_val, new_key, new_val, mask};

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
    key_concat->set_friendly_name("concat_key.0");
    auto val_concat = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 2);
    val_concat->set_friendly_name("concat_value.0");

    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    add->set_friendly_name("add.0");
    auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
    softmax->set_friendly_name("softmax.0");
    auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), val_concat->output(0));
    matmul2->set_friendly_name("matmul2.0");

    auto make_result = [&](const Output<Node>& out, const std::string& name) {
        results.push_back(std::make_shared<op::v0::Result>(out));
        results.back()->set_friendly_name(name);
    };
    make_result(key_concat->output(0), "present.0.key");
    make_result(val_concat->output(0), "present.0.value");
    make_result(matmul2->output(0), "attn_out.0");

    auto model = std::make_shared<Model>(results, params, "sdpa_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Build a model where Q is f32 but K/V cache are f16 (Gemma-4 style mixed precision).
std::shared_ptr<ov::Model> build_sdpa_model_mixed_dtype(size_t query_size = QUERY_SIZE,
                                                        size_t past_len = PAST_LEN,
                                                        size_t num_heads = NUM_HEADS,
                                                        size_t head_dim = HEAD_DIM) {
    using namespace ov;
    const size_t context_size = past_len + query_size;
    const Shape kv_shape = {BATCH, num_heads, past_len, head_dim};
    const Shape new_kv_shape = {BATCH, num_heads, query_size, head_dim};
    const Shape q_shape_s = {BATCH, num_heads, query_size, head_dim};
    const Shape mask_shape = {BATCH, 1, query_size, context_size};

    ResultVector results;

    // Q is f32 (compute precision), KV cache stored as f16 (storage precision),
    // present-KV from the upstream NPU subgraph is f32.
    // This mirrors the real Gemma-4 pattern:
    //   Convert(f16 past_block) ─┐
    //   f32 present_kv           ┴→ Concat(f32) → MatMul
    auto query = ov::test::utils::make_param(element::f32, q_shape_s, "query.0");
    auto past_key = ov::test::utils::make_param(element::f16, kv_shape, "past_key_values.0.key");
    auto past_val = ov::test::utils::make_param(element::f16, kv_shape, "past_key_values.0.value");
    auto new_key = ov::test::utils::make_param(element::f32, new_kv_shape, "new_key.0");
    auto new_val = ov::test::utils::make_param(element::f32, new_kv_shape, "new_value.0");
    auto mask = ov::test::utils::make_param(element::f32, mask_shape, "mask.0");
    ParameterVector params = {query, past_key, past_val, new_key, new_val, mask};

    // Upcast stored f16 KV blocks before Concat (matches block_kv_dtype derivation).
    auto past_key_f32 = std::make_shared<op::v0::Convert>(past_key, element::f32);
    auto past_val_f32 = std::make_shared<op::v0::Convert>(past_val, element::f32);

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key_f32, new_key}, 2);
    key_concat->set_friendly_name("concat_key.0");
    auto val_concat = std::make_shared<op::v0::Concat>(OutputVector{past_val_f32, new_val}, 2);
    val_concat->set_friendly_name("concat_value.0");

    // Q@K and softmax@V — both sides already f32, no extra Convert needed.
    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    add->set_friendly_name("add.0");
    auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
    softmax->set_friendly_name("softmax.0");
    auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), val_concat);
    matmul2->set_friendly_name("matmul2.0");

    auto make_result = [&](const Output<Node>& out, const std::string& name) {
        results.push_back(std::make_shared<op::v0::Result>(out));
        results.back()->set_friendly_name(name);
    };
    make_result(key_concat->output(0), "present.0.key");
    make_result(val_concat->output(0), "present.0.value");
    make_result(matmul2->output(0), "attn_out.0");

    auto model = std::make_shared<Model>(results, params, "sdpa_model_mixed_dtype");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

// ============================================================================
// MixedDtype suite — Q=f32, KV=f16.  The tile model must declare the Q
// parameter as f32 and KV/state parameters as f16, matching runtime tensors.
// ============================================================================

TEST(HostFlashAttentionMixedDtypeTest, FromReturnsValue) {
    EXPECT_TRUE(ov::npuw::function::HostFlashAttention::from(build_sdpa_model_mixed_dtype(), true).has_value());
}

// Helper: get element type of a model input by HFATileInputId name
static ov::element::Type get_input_dtype(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& in : model->inputs()) {
        if (in.get_names().count(name))
            return in.get_element_type();
    }
    throw std::runtime_error("input '" + name + "' not found in model");
}

// Q parameter must be f32 (matching query_tensor dtype at runtime)
TEST(HostFlashAttentionMixedDtypeTest, Fused_QParamIsF32) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_mixed_dtype(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(get_input_dtype(result->_tile_model, "Q"), ov::element::f32)
        << "Q tile parameter must match query tensor dtype (f32)";
    EXPECT_EQ(get_input_dtype(result->_final_tile_model, "Q"), ov::element::f32);
}

// KV tile parameters: regular tile = f16 (KV block storage), final tile = f32 (present-KV).
// This is the core invariant of the mixed-dtype fix.
TEST(HostFlashAttentionMixedDtypeTest, Fused_KVTileParamsAreF16) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_mixed_dtype(), true);
    ASSERT_TRUE(result.has_value());
    // Regular tile reads stored KV blocks → f16.
    EXPECT_EQ(get_input_dtype(result->_tile_model, "K_TILE"), ov::element::f16)
        << "K_TILE parameter must match KV cache dtype (f16)";
    EXPECT_EQ(get_input_dtype(result->_tile_model, "V_TILE"), ov::element::f16)
        << "V_TILE parameter must match KV cache dtype (f16)";
    // Final tile receives present-KV from upstream NPU subgraph → f32.
    EXPECT_EQ(get_input_dtype(result->_final_tile_model, "K_TILE"), ov::element::f32)
        << "Final tile K_TILE must match present-KV dtype (f32)";
    EXPECT_EQ(get_input_dtype(result->_final_tile_model, "V_TILE"), ov::element::f32)
        << "Final tile V_TILE must match present-KV dtype (f32)";
}

// State parameters: both tile models use f16 so regular-tile outputs feed final-tile
// inputs without conversion.
TEST(HostFlashAttentionMixedDtypeTest, Fused_StateParamsAreF16) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_mixed_dtype(), true);
    ASSERT_TRUE(result.has_value());
    for (const auto& tile_model : {result->_tile_model, result->_final_tile_model}) {
        EXPECT_EQ(get_input_dtype(tile_model, "PAST_ACC"), ov::element::f16);
        EXPECT_EQ(get_input_dtype(tile_model, "PAST_MAX"), ov::element::f16);
        EXPECT_EQ(get_input_dtype(tile_model, "PAST_D"), ov::element::f16);
    }
}

namespace {

void expect_input_name(const std::shared_ptr<ov::Model>& model,
                       size_t idx,
                       const std::string& expected_name,
                       const char* context) {
    ASSERT_LT(idx, model->inputs().size()) << context << ": model has too few inputs";
    const auto& names = model->inputs()[idx].get_names();
    EXPECT_TRUE(names.count(expected_name) > 0) << context << ": expected \"" << expected_name << "\" at index " << idx
                                                << ", got: " << (names.empty() ? "(none)" : *names.begin());
}

void check_input_shapes(const std::shared_ptr<ov::Model>& model,
                        const std::vector<ov::Shape>& expected,
                        const char* ctx) {
    ASSERT_EQ(model->inputs().size(), expected.size()) << ctx << ": unexpected number of inputs";
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(model->inputs()[i].get_shape(), expected[i])
            << ctx << ": shape mismatch at input[" << i << "] (" << model->inputs()[i].get_any_name() << ")";
    }
}

void check_output_shapes(const std::shared_ptr<ov::Model>& model,
                         const std::vector<ov::Shape>& expected,
                         const char* ctx) {
    ASSERT_EQ(model->outputs().size(), expected.size()) << ctx << ": unexpected number of outputs";
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(model->outputs()[i].get_shape(), expected[i]) << ctx << ": shape mismatch at output[" << i << "]";
    }
}

// Test models built in this file don't run DetectAttentionMask, so their Add(QK, mask)
// node starts unannotated (Unknown). This helper emulates what DetectAttentionMask would
// have written, so tests can exercise HostFlashAttention::from()'s per-SDPA mask-skipping
// decision directly. `encoded_value` follows NPUW_SDPA_MASK_RT_KEY's encoding: negative
// (e.g. ov::npuw::NPUW_SDPA_MASK_CAUSAL) for Causal, >= 0 for SlidingWindow(window_size).
void annotate_mask_rt_info(const std::shared_ptr<ov::Model>& model,
                           int64_t encoded_value,
                           const std::string& add_name = "add.0") {
    for (const auto& node : model->get_ops()) {
        auto add = ov::as_type_ptr<ov::op::v1::Add>(node);
        if (add && add->get_friendly_name() == add_name) {
            add->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = encoded_value;
            return;
        }
    }
    throw std::runtime_error("add node '" + add_name + "' not found");
}

}  // namespace

TEST(HostFlashAttentionFromTest, ReturnsNulloptForNonSDPAModel) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{BATCH, NUM_HEADS * HEAD_DIM});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(param)},
                                             ov::ParameterVector{param},
                                             "plain_model");
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, false).has_value());
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, true).has_value());
}

TEST(HostFlashAttentionFromTest, NonFused_FinalTileHasSevenInputs) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
}

// ============================================================================
// Parameter index consistency: indices 0-5 identical in both tile models
// ============================================================================

TEST(HostFlashAttentionFromTest, NonFused_TileInputNamesMatchAtIndicesZeroToFive) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    const auto& reg = result->_tile_model->inputs();
    const auto& fin = result->_final_tile_model->inputs();
    ASSERT_GE(reg.size(), 6u);
    ASSERT_GE(fin.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(reg[i].get_names(), fin[i].get_names()) << "mismatch at index " << i;
    }
}

TEST(HostFlashAttentionFromTest, Fused_TileInputNamesMatchAtIndicesZeroToFive) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    const auto& reg = result->_tile_model->inputs();
    const auto& fin = result->_final_tile_model->inputs();
    ASSERT_GE(reg.size(), 6u);
    ASSERT_GE(fin.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(reg[i].get_names(), fin[i].get_names()) << "mismatch at index " << i;
    }
}

// ============================================================================
// Mask tensor is at index 6 with correct name
// ============================================================================

TEST(HostFlashAttentionFromTest, NonFused_MaskTileAtIndexSixInBothModels) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    expect_input_name(result->_tile_model, 6, "MASK_TILE", "non-fused regular tile");
    expect_input_name(result->_final_tile_model, 6, "MASK_TILE", "non-fused final tile");
}

TEST(HostFlashAttentionFromTest, Fused_NoRtInfoAnnotation_KeepsMaskEvenWhenGlobalYes) {
    // Without a DetectAttentionMask annotation (Unknown), mask skipping must stay
    // disabled even when the global switch is on -- the mask shape/semantics are
    // unproven, so skipping it could silently change results.
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_model->inputs().size(), 7u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
    expect_input_name(result->_tile_model, 6, "MASK_TILE", "fused regular tile without rt_info annotation");
    expect_input_name(result->_final_tile_model, 6, "MASK_TILE", "fused final tile");
}

TEST(HostFlashAttentionFromTest, Fused_MaskTileAtIndexSixInFinalTileOnly) {
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);

    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_model->inputs().size(), 6u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
    expect_input_name(result->_final_tile_model, 6, "MASK_TILE", "fused final tile");
}

TEST(HostFlashAttentionFromTest, Fused_MaskTileAtIndexSixInRegularTileWhenMaskSkippingDisabled) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_model->inputs().size(), 7u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
    expect_input_name(result->_tile_model, 6, "MASK_TILE", "fused regular tile with mask skipping disabled");
    expect_input_name(result->_final_tile_model, 6, "MASK_TILE", "fused final tile");
}

TEST(HostFlashAttentionFromTest, Fused_PerSDPACausalRtInfo_EnablesRegularTileMaskSkipping) {
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);

    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());

    // Regular tile skips mask (6 inputs), final tile still keeps mask (7 inputs).
    EXPECT_EQ(result->_tile_model->inputs().size(), 6u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
}

TEST(HostFlashAttentionFromTest, Fused_PerSDPACausalRtInfo_DisabledByGlobalKillSwitch) {
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);

    // NPUW_ATTN_HFA_MASK_SKIPPING=NO (global) acts as a master kill switch: even a
    // Causal per-SDPA annotation cannot re-enable mask skipping.
    auto result = ov::npuw::function::HostFlashAttention::from(model, true, false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_model->inputs().size(), 7u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
}

TEST(HostFlashAttentionFromTest, Fused_PerSDPASlidingRtInfo_NarrowerThanContext_KeepsMask) {
    auto model = build_sdpa_model();  // context_size = QUERY_SIZE + PAST_LEN
    annotate_mask_rt_info(model, /*window_size=*/8);

    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());

    // Window is narrower than the context, so a regular tile can't safely skip the
    // mask -- keeps mask (7 inputs) in both tiles.
    EXPECT_EQ(result->_tile_model->inputs().size(), 7u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
}

TEST(HostFlashAttentionFromTest, Fused_PerSDPASlidingRtInfo_CoversWholeContext_SkipsRegularMask) {
    auto model = build_sdpa_model();  // context_size = QUERY_SIZE + PAST_LEN
    annotate_mask_rt_info(model, /*window_size=*/static_cast<int64_t>(QUERY_SIZE + PAST_LEN));

    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());

    // Window covers the whole context, behaves like Causal -- skips mask (6 inputs)
    // in the regular tile, final tile still keeps mask (7 inputs).
    EXPECT_EQ(result->_tile_model->inputs().size(), 6u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
}

// ============================================================================
// Tile param index map
// ============================================================================

TEST(HostFlashAttentionFromTest, NonFused_ParamIndexMapHasAllSevenEntries) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_param_index_map.size(), static_cast<size_t>(ov::npuw::HFATileInputId::COUNT));
}

TEST(HostFlashAttentionFromTest, Fused_ParamIndexMapHasAllSevenEntries) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_param_index_map.size(), static_cast<size_t>(ov::npuw::HFATileInputId::COUNT));
}

TEST(HostFlashAttentionFromTest, Fused_MaskTileIndexInMapIsSix) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    auto it = result->_tile_param_index_map.find(ov::npuw::HFATileInputId::MASK_TILE);
    ASSERT_NE(it, result->_tile_param_index_map.end());
    EXPECT_EQ(it->second, 6u);
}

TEST(HostFlashAttentionFromTest, Fused_TileSizeAndQuerySizeAreCorrect) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_past_tile_size, static_cast<int64_t>(QUERY_SIZE));
    EXPECT_EQ(result->_final_tile_size, static_cast<int64_t>(QUERY_SIZE));
    EXPECT_EQ(result->_query_size, QUERY_SIZE);
}

TEST(HostFlashAttentionFromTest, Fused_ContextSizeIsCorrect) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_context_size, QUERY_SIZE + PAST_LEN);
}

// ============================================================================
// analyze_past_tiling (past_tile_size / final_tile_size split for SWA support)
// ============================================================================

// SWA-shrunk past (past_len < query_size): the single past input has nothing to chunk
// with, so past_tile_size takes on its own (shorter) length instead of query_size.
TEST(HostFlashAttentionFromTest, Fused_ShortPastYieldsPastTileSizeEqualToPastLen) {
    constexpr size_t short_past_len = QUERY_SIZE / 2;
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(QUERY_SIZE, short_past_len), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_past_tile_size, static_cast<int64_t>(short_past_len));
    EXPECT_EQ(result->_final_tile_size, static_cast<int64_t>(QUERY_SIZE));
}

// Past length that is an exact multiple of query_size (block-split/continuous, non-SWA):
// past_tile_size stays equal to query_size regardless of how many multiples the past holds.
TEST(HostFlashAttentionFromTest, Fused_PastMultipleOfQuerySizeYieldsPastTileSizeEqualToQuerySize) {
    constexpr size_t multi_past_len = QUERY_SIZE * 3;
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(QUERY_SIZE, multi_past_len), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_past_tile_size, static_cast<int64_t>(QUERY_SIZE));
    EXPECT_EQ(result->_final_tile_size, static_cast<int64_t>(QUERY_SIZE));
}

// Past length >= query_size but not an exact multiple of it: analyze_past_tiling must
// reject this (PREFILL is expected to fill the KV cache in exact query_size increments).
TEST(HostFlashAttentionFromTest, Fused_PastNotMultipleOfQuerySizeThrows) {
    constexpr size_t bad_past_len = QUERY_SIZE + QUERY_SIZE / 2;
    EXPECT_THROW(ov::npuw::function::HostFlashAttention::from(build_sdpa_model(QUERY_SIZE, bad_past_len), true),
                 ov::Exception);
}

namespace {
// ============================================================================
// Build a model where V is pre-transposed (axis=3), simulating a model that
// has been processed by OptimizeValueTensors.  V parameters are stored as
// [B, H, head_dim, seq] and the V-Concat is along axis 3.
// ============================================================================
std::shared_ptr<ov::Model> build_sdpa_model_transposed_v(size_t query_size = QUERY_SIZE,
                                                         size_t past_len = PAST_LEN,
                                                         size_t num_heads = NUM_HEADS,
                                                         size_t head_dim = HEAD_DIM) {
    using namespace ov;
    const Shape past_k_shape = {BATCH, num_heads, past_len, head_dim};  // K: normal layout
    const Shape new_k_shape = {BATCH, num_heads, query_size, head_dim};
    const Shape past_v_shape = {BATCH, num_heads, head_dim, past_len};  // V: pre-transposed
    const Shape new_v_shape = {BATCH, num_heads, head_dim, query_size};
    const Shape q_shape = {BATCH, num_heads, query_size, head_dim};
    const Shape mask_shape = {BATCH, 1, query_size, past_len + query_size};

    ResultVector results;
    auto query = ov::test::utils::make_param(element::f32, q_shape, "query.0");
    auto past_key = ov::test::utils::make_param(element::f32, past_k_shape, "past_key_values.0.key");
    auto past_val = ov::test::utils::make_param(element::f32, past_v_shape, "past_key_values.0.value");
    auto new_key = ov::test::utils::make_param(element::f32, new_k_shape, "new_key.0");
    auto new_val = ov::test::utils::make_param(element::f32, new_v_shape, "new_value.0");
    auto mask = ov::test::utils::make_param(element::f32, mask_shape, "mask.0");
    ParameterVector params = {query, past_key, past_val, new_key, new_val, mask};

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
    key_concat->set_friendly_name("concat_key.0");
    // axis=3: concat along the last dim, which is the sequence dimension in transposed V layout
    auto val_concat = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 3);
    val_concat->set_friendly_name("concat_value.0");

    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    add->set_friendly_name("add.0");
    auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
    softmax->set_friendly_name("softmax.0");
    // transpose_b=true: softmax[B,H,q,k] x V[B,H,head_dim,k]^T -> [B,H,q,head_dim]
    auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), val_concat->output(0), false, true);
    matmul2->set_friendly_name("matmul2.0");

    auto make_result = [&](const Output<Node>& out, const std::string& name) {
        results.push_back(std::make_shared<op::v0::Result>(out));
        results.back()->set_friendly_name(name);
    };
    make_result(key_concat->output(0), "present.0.key");
    make_result(val_concat->output(0), "present.0.value");
    make_result(matmul2->output(0), "attn_out.0");

    auto model = std::make_shared<Model>(results, params, "sdpa_model_transposed_v");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

// ============================================================================
// Input / output shape checks — V NOT transposed (axis=2, the default)
// Expected shapes (BATCH=1, NUM_HEADS=8, HEAD_DIM=64, QUERY_SIZE=16):
//   past_acc  [1,8,16,64]  past_max [1,8,16,1]  past_d [1,8,16,1]
//   k_tile    [1,8,16,64]  v_tile   [1,8,16,64]  (normal [B,H,tile,head_dim])
//   q         [1,8,16,64]  mask_tile [1,1,16,16]
// ============================================================================

// Fused path with mask skipping enabled — regular tile (6 inputs, no mask); v_tile in normal layout
TEST(HostFlashAttentionFromTest, Fused_RegularTileInputShapes) {
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile  [B, kv_heads, tile, head_dim]
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // v_tile  [B, kv_heads, tile, head_dim] (normal)
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
    };
    check_input_shapes(result->_tile_model, expected_inputs, "fused regular tile");
}

// Fused path — final tile (7 inputs, with mask); v_tile in normal layout
TEST(HostFlashAttentionFromTest, Fused_FinalTileInputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // v_tile (normal)
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
        {BATCH, 1, QUERY_SIZE, QUERY_SIZE},        // mask_tile [B, 1, seq, tile]
    };
    check_input_shapes(result->_final_tile_model, expected_inputs, "fused final tile");
}

// Non-fused path — regular tile (7 inputs, with mask); v_tile in normal layout
TEST(HostFlashAttentionFromTest, NonFused_RegularTileInputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // v_tile (normal)
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
        {BATCH, 1, QUERY_SIZE, QUERY_SIZE},        // mask_tile
    };
    check_input_shapes(result->_tile_model, expected_inputs, "non-fused regular tile");
}

// Regular tile outputs: acc [B,H,L,E]  maxx [B,H,L,1]  d [B,H,L,1]
TEST(HostFlashAttentionFromTest, Fused_RegularTileOutputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    const std::vector<ov::Shape> expected = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // maxx
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // d
    };
    check_output_shapes(result->_tile_model, expected, "fused regular tile");
}

TEST(HostFlashAttentionFromTest, NonFused_RegularTileOutputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    const std::vector<ov::Shape> expected = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // maxx
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // d
    };
    check_output_shapes(result->_tile_model, expected, "non-fused regular tile");
}

// Final tile output: [B, QUERY_SIZE, NUM_HEADS * HEAD_DIM] after transpose + reshape
TEST(HostFlashAttentionFromTest, Fused_FinalTileOutputShape) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    check_output_shapes(result->_final_tile_model, {{BATCH, QUERY_SIZE, NUM_HEADS * HEAD_DIM}}, "fused final tile");
}

TEST(HostFlashAttentionFromTest, NonFused_FinalTileOutputShape) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    check_output_shapes(result->_final_tile_model, {{BATCH, QUERY_SIZE, NUM_HEADS * HEAD_DIM}}, "non-fused final tile");
}

// ============================================================================
// TransposedV suite — V pre-transposed (axis=3), simulating OptimizeValueTensors.
// v_tile must be [B, H, head_dim, tile_size] (transposed storage layout).
// ============================================================================

// Sanity: the transposed-V model is parseable
TEST(HostFlashAttentionTransposedVTest, FromReturnsValue) {
    EXPECT_TRUE(ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true).has_value());
}

// v_seq_dim == 3 -> _v_seq_dim field stored correctly
TEST(HostFlashAttentionTransposedVTest, VSeqDimIsThree) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_v_seq_dim, 3u);
}

// Contrast: non-transposed model has _v_seq_dim == 2
TEST(HostFlashAttentionFromTest, VSeqDimIsTwoForNormalModel) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_v_seq_dim, 2u);
}

// Fused regular tile with mask skipping enabled: v_tile in transposed layout [B, H, head_dim, tile]
TEST(HostFlashAttentionTransposedVTest, Fused_RegularTileVTileIsTransposed) {
    auto model = build_sdpa_model_transposed_v();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);
    ASSERT_TRUE(result.has_value());
    const std::vector<ov::Shape> expected = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile
        {BATCH, NUM_HEADS, HEAD_DIM, QUERY_SIZE},  // v_tile [B,H,head_dim,tile] — pre-transposed
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
    };
    check_input_shapes(result->_tile_model, expected, "transposed-V fused regular tile");
}

// Fused final tile: v_tile in transposed layout [B, H, head_dim, tile]
TEST(HostFlashAttentionTransposedVTest, Fused_FinalTileVTileIsTransposed) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true);
    ASSERT_TRUE(result.has_value());
    const std::vector<ov::Shape> expected = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile
        {BATCH, NUM_HEADS, HEAD_DIM, QUERY_SIZE},  // v_tile [B,H,head_dim,tile] — pre-transposed
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
        {BATCH, 1, QUERY_SIZE, QUERY_SIZE},        // mask_tile
    };
    check_input_shapes(result->_final_tile_model, expected, "transposed-V fused final tile");
}

// Output shapes are independent of V layout
TEST(HostFlashAttentionTransposedVTest, Fused_FinalTileOutputShape) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true);
    ASSERT_TRUE(result.has_value());
    check_output_shapes(result->_final_tile_model,
                        {{BATCH, QUERY_SIZE, NUM_HEADS * HEAD_DIM}},
                        "transposed-V fused final tile");
}

TEST(HostFlashAttentionTransposedVTest, Fused_RegularTileOutputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true);
    ASSERT_TRUE(result.has_value());
    const std::vector<ov::Shape> expected = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},
    };
    check_output_shapes(result->_tile_model, expected, "transposed-V fused regular tile");
}

// Context size is derived from the transposed concat (axis=3, dim=3 of output)
TEST(HostFlashAttentionTransposedVTest, ContextSizeIsCorrect) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model_transposed_v(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_context_size, QUERY_SIZE + PAST_LEN);
}

// ---------------------------------------------------------------------------
// runtime::host_flash_attention::PositionIDs::find() shape matching
// ---------------------------------------------------------------------------
namespace {

// Minimal IPlugin stub needed to construct an ICompiledModel.
class PosIdsNullPlugin final : public ov::IPlugin {
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

// ICompiledModel stub that just exposes the wrapped model's inputs/outputs.
class PosIdsStubCompiledModel final : public ov::ICompiledModel {
public:
    PosIdsStubCompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin)
        : ov::ICompiledModel(model, plugin) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return nullptr;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        return nullptr;
    }
};

// Minimal ISyncInferRequest with tensor binding for selector preparation.
class PosIdsFakeInferRequest final : public ov::ISyncInferRequest {
public:
    explicit PosIdsFakeInferRequest(const std::shared_ptr<const ov::ICompiledModel>& cm) : ov::ISyncInferRequest(cm) {}

    void infer() override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    void check_tensors() const override {}
};

// Builds a model with a single "position_ids" parameter of the given shape (plus a dummy
// output so the model is valid) and wraps it into a fake infer request usable by find().
std::shared_ptr<ov::ISyncInferRequest> make_position_ids_request(const ov::Shape& position_ids_shape) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, position_ids_shape);
    position_ids->set_friendly_name("position_ids");
    position_ids->output(0).get_tensor().set_names({"position_ids"});
    auto result = std::make_shared<ov::op::v0::Result>(position_ids);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{position_ids});

    auto plugin = std::make_shared<PosIdsNullPlugin>();
    auto compiled = std::make_shared<PosIdsStubCompiledModel>(model, plugin);
    return std::make_shared<PosIdsFakeInferRequest>(compiled);
}

void bind_position_ids(const std::shared_ptr<ov::ISyncInferRequest>& request, std::initializer_list<int64_t> values) {
    const auto& port = request->get_inputs().front();
    ov::Tensor tensor(ov::element::i64, port.get_shape());
    ASSERT_EQ(tensor.get_size(), values.size());
    std::copy(values.begin(), values.end(), tensor.data<int64_t>());
    request->set_tensor(port, ov::get_tensor_impl(tensor));
}

}  // namespace

TEST(HostFlashAttentionPositionIdsFindTest, Matches1D) {
    auto rq = make_position_ids_request(ov::Shape{16});
    EXPECT_NE(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionPositionIdsFindTest, Matches2D) {
    auto rq = make_position_ids_request(ov::Shape{1, 16});
    EXPECT_NE(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

// Qwen2.5-VL mrope layout: [3, 1, seq_len].
TEST(HostFlashAttentionPositionIdsFindTest, Matches3DMropeThreeSections) {
    auto rq = make_position_ids_request(ov::Shape{3, 1, 16});
    EXPECT_NE(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

// Qwen3.5-VL mrope layout: [4, 1, seq_len].
TEST(HostFlashAttentionPositionIdsFindTest, Matches3DMropeFourSections) {
    auto rq = make_position_ids_request(ov::Shape{4, 1, 16});
    EXPECT_NE(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionPositionIdsFindTest, RejectsNonUnitBatchDim) {
    auto rq = make_position_ids_request(ov::Shape{2, 16});
    EXPECT_EQ(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionPositionIdsFindTest, RejectsNonUnitBatchDimIn3D) {
    auto rq = make_position_ids_request(ov::Shape{3, 2, 16});
    EXPECT_EQ(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionPositionIdsFindTest, HostWindowAcceptsFullCacheWithBaselineOneBelowWindow) {
    auto rq = make_position_ids_request({1, 8});
    bind_position_ids(rq, {17, 18, 19, 20, 21, 22, 23, 24});
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq, 25);
    ASSERT_NE(selector, nullptr);
    EXPECT_NO_THROW(selector->prepare(16));
    EXPECT_EQ(selector->context_length(), 24);
}

TEST(HostFlashAttentionPositionIdsFindTest, HostWindowRejectsFullCacheAtWindowWithDisableInstruction) {
    for (int64_t window : {24, 25}) {
        SCOPED_TRACE(window);
        auto rq = make_position_ids_request({1, 8});
        bind_position_ids(rq,
                          {window - 7, window - 6, window - 5, window - 4, window - 3, window - 2, window - 1, window});
        auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq, window);
        ASSERT_NE(selector, nullptr);
        try {
            selector->prepare(16);
            FAIL() << "A position at the F2 window must reject host-mask execution";
        } catch (const ov::Exception& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("NPUW_ATTN_HFA_HOST_MASK"), std::string::npos);
            EXPECT_NE(message.find("NPUW_ATTN_HFA_HOST_MASK=NO"), std::string::npos);
            EXPECT_NE(message.find(std::to_string(window)), std::string::npos);
            auto lower = message;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            EXPECT_NE(lower.find("window"), std::string::npos);
            EXPECT_NE(lower.find("position"), std::string::npos);
            EXPECT_NE(lower.find("disable"), std::string::npos);
        }
    }
}

TEST(HostFlashAttentionPositionIdsFindTest, HostWindowChecksMaximumNotLastPosition) {
    for (int64_t maximum : {25, 26}) {
        SCOPED_TRACE(maximum);
        auto rq = make_position_ids_request({1, 8});
        bind_position_ids(rq, {17, 18, 19, maximum, 21, 22, 23, 24});
        auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq, 25);
        ASSERT_NE(selector, nullptr);
        EXPECT_THROW(selector->prepare(16), ov::Exception);
    }
}

TEST(HostFlashAttentionPositionIdsFindTest, HostWindowRechecksUpdatedTensorForEachChunk) {
    auto rq = make_position_ids_request({1, 8});
    bind_position_ids(rq, {9, 10, 11, 12, 13, 14, 15, 16});
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq, 25);
    ASSERT_NE(selector, nullptr);
    EXPECT_NO_THROW(selector->prepare(8));
    EXPECT_EQ(selector->context_length(), 16);
    bind_position_ids(rq, {18, 19, 20, 21, 22, 23, 24, 25});
    EXPECT_THROW(selector->prepare(16), ov::Exception);
}

TEST(HostFlashAttentionPositionIdsFindTest, ZeroWindowKeepsLegacyPositionDomain) {
    auto rq = make_position_ids_request({1, 8});
    bind_position_ids(rq, {18, 19, 20, 21, 22, 23, 24, 25});
    auto legacy = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq);
    auto unrestricted = ov::npuw::runtime::host_flash_attention::PositionIDs::find(8, *rq, 0);
    ASSERT_NE(legacy, nullptr);
    ASSERT_NE(unrestricted, nullptr);
    EXPECT_NO_THROW(legacy->prepare(16));
    EXPECT_NO_THROW(unrestricted->prepare(16));
    EXPECT_EQ(legacy->context_length(), 24);
    EXPECT_EQ(unrestricted->context_length(), legacy->context_length());
}

namespace {
namespace hfa = ov::npuw::runtime::host_flash_attention;

ov::Tensor make_host_row(size_t context, size_t history = 0) {
    ov::Tensor row(ov::element::i64, {1, context});
    std::fill_n(row.data<int64_t>(), context, 0);
    std::fill_n(row.data<int64_t>(), history, 1);
    return row;
}

template <typename T>
void check_host_tile_reference(size_t query_size,
                               size_t real_tokens,
                               size_t physical_rows,
                               bool final,
                               size_t history = 13) {
    constexpr size_t context = 48;
    auto row = make_host_row(context, history);
    std::fill_n(row.data<int64_t>() + context - real_tokens, real_tokens, 1);
    const size_t offset = final ? context - query_size : 8;
    const size_t columns = final ? query_size : 8;
    const auto dtype = ov::element::from<T>();
    auto key = hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row),
                                            offset,
                                            columns,
                                            query_size,
                                            physical_rows,
                                            final,
                                            dtype,
                                            history != 0);
    ov::Tensor tile(dtype, {1, 1, physical_rows, columns});
    hfa::fill_host_mask_tile(key, ov::get_tensor_impl(tile));
    // Build the full reference independently, then select the physical query suffix.
    std::vector<T> reference(query_size * context);
    for (size_t r = 0; r < query_size; ++r) {
        bool row_visible = false;
        for (size_t c = 0; c < context; ++c) {
            const bool in_past = c < 16;
            const bool in_present = c >= context - query_size && c - (context - query_size) <= r;
            const bool visible = row.data<int64_t>()[c] != 0 && (in_past || in_present);
            row_visible |= visible;
            reference[r * context + c] = visible ? T(0) : std::numeric_limits<T>::lowest();
        }
        if (!row_visible) {
            for (size_t c = context - query_size; c < context; ++c) {
                reference[r * context + c] = T(0);
            }
        }
    }
    for (size_t r = 0; r < physical_rows; ++r) {
        for (size_t c = 0; c < columns; ++c) {
            const auto value = tile.data<T>()[r * columns + c];
            EXPECT_EQ(value, reference[(query_size - physical_rows + r) * context + offset + c]);
            EXPECT_TRUE(std::isfinite(static_cast<float>(value)));
        }
    }
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> make_host_pass_model(
    std::shared_ptr<ov::Model>& model,
    int64_t annotation,
    ov::test::npuw::host_mask_catalogue::Family family = ov::test::npuw::host_mask_catalogue::Family::F1,
    ov::test::npuw::host_mask_catalogue::Layout layout = ov::test::npuw::host_mask_catalogue::Layout::Direct,
    bool inputs_embeds = false,
    ov::test::npuw::host_mask_catalogue::Mask* cone = nullptr) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    auto q = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 8, 4});
    auto past = ov::test::utils::make_param(element::f32, Shape{1, 2, 16, 4}, "past_key_values.0.key");
    auto k = std::make_shared<op::v0::Concat>(OutputVector{past, q}, 2);
    auto v = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 24, 4});
    auto row = ov::test::utils::make_param(element::i64, Shape{1, 24}, "attention_mask");
    Output<Node> sequence = std::make_shared<op::v1::Reshape>(q, catalogue::vector({1, 8, 8}), false);
    ParameterVector parameters{q, past, v, row};
    if (inputs_embeds) {
        auto embeds = ov::test::utils::make_param(element::f32, Shape{1, 8, 8}, "inputs_embeds");
        parameters.push_back(embeds);
        sequence = embeds;
    }
    Output<Node> positions = sequence;
    if (family == catalogue::Family::F2) {
        auto pos = ov::test::utils::make_param(element::i64, Shape{1, 8}, "position_ids");
        parameters.push_back(pos);
        positions = pos;
        if (layout == catalogue::Layout::Phi4) {
            sequence = pos;
        }
    }
    auto mask = catalogue::make_mask(family, row, past, sequence, positions, annotation, layout);
    if (cone) {
        *cone = mask;
    }
    Output<Node> physical_q = q;
    if (layout == catalogue::Layout::QuerySuffix) {
        physical_q = std::make_shared<op::v8::Slice>(q,
                                                     catalogue::vector({7}),
                                                     catalogue::vector({8}),
                                                     catalogue::vector({1}),
                                                     catalogue::vector({2}));
    }
    auto sdpa = std::make_shared<op::v13::ScaledDotProductAttention>(physical_q, k, v, mask.root, false);
    sdpa->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = annotation;
    model = std::make_shared<Model>(OutputVector{sdpa}, parameters);
    return sdpa;
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> make_detected_host_pass_model(std::shared_ptr<ov::Model>& model,
                                                                                      bool image_blocks,
                                                                                      int64_t window = 0,
                                                                                      bool inputs_embeds = false) {
    using namespace ov;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    auto row = model->get_parameters().back();
    auto seq = ov::test::utils::make_param(inputs_embeds ? element::f32 : element::i64,
                                           inputs_embeds ? Shape{1, 8, 4} : Shape{1, 8},
                                           inputs_embeds ? "inputs_embeds" : "input_ids");
    model->add_parameters({seq});
    const auto causal = ov::test::npuw::make_causal_bool(seq, row, "host_mask_test.");
    Output<Node> visible = causal.mask;
    if (window > 0) {
        auto bound =
            std::make_shared<op::v1::Add>(causal.q_col, op::v0::Constant::create(element::i64, Shape{}, {-window}));
        auto in_window = std::make_shared<op::v1::Greater>(causal.kv_row, bound);
        auto padding = std::make_shared<op::v0::Convert>(row, element::boolean);
        auto window_padding = std::make_shared<op::v13::BitwiseAnd>(padding, in_window);
        visible = std::make_shared<op::v13::BitwiseAnd>(window_padding, visible);
    }
    if (image_blocks) {
        auto types = ov::test::utils::make_param(element::i64, Shape{1, 24}, "token_type_ids");
        model->add_parameters({types});
        auto image_columns =
            std::make_shared<op::v1::Equal>(types, op::v0::Constant::create(element::i64, Shape{}, {1}));
        visible = std::make_shared<op::v13::BitwiseOr>(visible, image_columns);
    }
    auto padding = std::make_shared<op::v0::Convert>(row, element::boolean);
    visible = std::make_shared<op::v13::BitwiseAnd>(visible, padding);
    auto mask = std::make_shared<op::v1::Select>(
        visible,
        op::v0::Constant::create(element::f32, Shape{}, {0}),
        op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()}));
    auto mask_4d = std::make_shared<op::v0::Unsqueeze>(mask, op::v0::Constant::create(element::i64, Shape{2}, {0, 1}));
    sdpa->input(3).replace_source_output(mask_4d);
    model->validate_nodes_and_infer_types();
    npuw::DetectAttentionMask().run_on_model(model);
    return sdpa;
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> make_legacy_phi_host_pass_model(
    std::shared_ptr<ov::Model>& model,
    int64_t window,
    bool image_blocks = false,
    bool inputs_embeds = false) {
    using namespace ov;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    auto row = model->get_parameters().back();
    auto seq = ov::test::utils::make_param(inputs_embeds ? element::f32 : element::i64,
                                           inputs_embeds ? Shape{1, 8, 4} : Shape{1, 8},
                                           inputs_embeds ? "inputs_embeds" : "input_ids");
    model->add_parameters({seq});
    Output<Node> mask = ov::test::npuw::make_sliding_window_mask_phi3_legacy(seq, row, element::f32, window);
    if (image_blocks) {
        auto types = ov::test::utils::make_param(element::i64, Shape{1, 24}, "token_type_ids");
        model->add_parameters({types});
        auto image_columns =
            std::make_shared<op::v1::Equal>(types, op::v0::Constant::create(element::i64, Shape{}, {1}));
        auto visible = std::make_shared<op::v1::LogicalNot>(mask.get_node_shared_ptr()->input_value(0));
        auto with_images = std::make_shared<op::v13::BitwiseOr>(visible, image_columns);
        mask = std::make_shared<op::v1::Select>(
            with_images,
            op::v0::Constant::create(element::f32, Shape{}, {0}),
            op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()}));
    }
    auto row_float = std::make_shared<op::v0::Convert>(row, element::f32);
    auto padding = std::make_shared<op::v1::Equal>(std::make_shared<op::v1::Add>(mask, row_float),
                                                   op::v0::Constant::create(element::f32, Shape{}, {0}));
    auto padded = std::make_shared<op::v1::Select>(
        padding,
        op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()}),
        mask);
    sdpa->input(3).replace_source_output(padded);
    model->validate_nodes_and_infer_types();
    npuw::DetectAttentionMask().run_on_model(model);
    return sdpa;
}

void expect_host_mask_rejected(const std::shared_ptr<ov::Model>& model,
                               const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa) {
    const auto original = sdpa->input_value(3);
    const auto causal = sdpa->get_causal();
    const auto annotation = sdpa->get_rt_info().find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    const bool annotated = annotation != sdpa->get_rt_info().end();
    const auto encoded = annotated ? annotation->second.as<int64_t>() : 0;
    EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_EQ(sdpa->get_causal(), causal);
    EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
    EXPECT_EQ(sdpa->get_rt_info().count(ov::npuw::NPUW_SDPA_MASK_RT_KEY), annotated ? 1u : 0u);
    if (annotated) {
        EXPECT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), encoded);
    }
}

void detect_catalogue_window(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa,
                             const ov::test::npuw::host_mask_catalogue::Mask& cone,
                             const std::shared_ptr<ov::Node>& restriction = nullptr) {
    using namespace ov;
    auto visible = std::make_shared<op::v13::BitwiseAnd>(
        std::make_shared<op::v13::BitwiseAnd>(op::v0::Constant::create(element::boolean, Shape{}, {true}),
                                              cone.row_bound),
        std::make_shared<op::v1::LessEqual>(cone.columns, cone.rows));
    if (restriction) {
        visible = std::make_shared<op::v13::BitwiseAnd>(visible, restriction);
    }
    const auto patched = sdpa->input_value(3);
    sdpa->input(3).replace_source_output(std::make_shared<op::v1::Select>(
        visible,
        op::v0::Constant::create(element::f32, Shape{}, {0}),
        op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()})));
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    // Detection runs on the exported window anchor before the prefill mask patch.
    npuw::DetectAttentionMask().run_on_model(model);
    sdpa->input(3).replace_source_output(patched);
}

void expect_catalogue_mask_replaced(
    ov::test::npuw::host_mask_catalogue::Family family,
    ov::test::npuw::host_mask_catalogue::Layout layout = ov::test::npuw::host_mask_catalogue::Layout::Direct,
    bool embeds = false) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    const int64_t annotation = family == catalogue::Family::F2 ? 262144 : ov::npuw::NPUW_SDPA_MASK_CAUSAL;
    std::shared_ptr<ov::Model> model;
    auto sdpa = make_host_pass_model(model, annotation, family, layout, embeds);
    EXPECT_EQ(sdpa->input_value(3).get_shape(),
              (ov::Shape{1, 1, layout == catalogue::Layout::QuerySuffix ? 1u : 8u, 24}));
    EXPECT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_TRUE(ov::npuw::has_hfa_host_mask(model));
    EXPECT_FALSE(sdpa->get_causal());
    auto convert = ov::as_type_ptr<ov::op::v0::Convert>(sdpa->get_input_node_shared_ptr(3));
    ASSERT_NE(convert, nullptr);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(convert->get_input_node_shared_ptr(0));
    ASSERT_NE(reshape, nullptr);
    EXPECT_EQ(reshape->get_shape(), (ov::Shape{1, 1, 1, 24}));
    EXPECT_EQ(reshape->get_input_node_shared_ptr(0), model->get_parameters()[3]);
}
}  // namespace

TEST(HFAHostMaskTest, AlignedHistoryAndFirstChunkMap) {
    auto row = make_host_row(64, 16);
    EXPECT_EQ(hfa::build_host_mask_map(*ov::get_tensor_impl(row), 16, 8),
              (std::vector<hfa::HostMaskTileAction>{hfa::HostMaskTileAction::NoMask, hfa::HostMaskTileAction::NoMask}));
    EXPECT_TRUE(hfa::build_host_mask_map(*ov::get_tensor_impl(row), 0, 8).empty());
}

TEST(HFAHostMaskTest, UnalignedHistoryGapMap) {
    // h=2*T+5=21, Q=16, P=32: tile 2 is partial and tile 3 is wholly in the gap.
    auto row = make_host_row(64, 21);
    EXPECT_EQ(hfa::build_host_mask_map(*ov::get_tensor_impl(row), 32, 8),
              (std::vector<hfa::HostMaskTileAction>{hfa::HostMaskTileAction::NoMask,
                                                    hfa::HostMaskTileAction::NoMask,
                                                    hfa::HostMaskTileAction::Mask,
                                                    hfa::HostMaskTileAction::Skip}));
}

TEST(HFAHostMaskTest, HistoryHoleMap) {
    auto row = make_host_row(64, 24);
    row.data<int64_t>()[11] = 0;
    EXPECT_EQ(hfa::build_host_mask_map(*ov::get_tensor_impl(row), 24, 8),
              (std::vector<hfa::HostMaskTileAction>{hfa::HostMaskTileAction::NoMask,
                                                    hfa::HostMaskTileAction::Mask,
                                                    hfa::HostMaskTileAction::NoMask}));
}

TEST(HFAHostMaskTest, FinalTilesMatchFullReferenceF32) {
    check_host_tile_reference<float>(8, 8, 8, true);
    check_host_tile_reference<float>(8, 3, 8, true);
    check_host_tile_reference<float>(8, 1, 8, true, 0);
}

TEST(HFAHostMaskTest, FinalTilesMatchFullReferenceF16) {
    check_host_tile_reference<ov::float16>(8, 8, 8, true);
    check_host_tile_reference<ov::float16>(8, 3, 8, true);
    check_host_tile_reference<ov::float16>(8, 1, 8, true, 0);
}

TEST(HFAHostMaskTest, SlicedQueryUsesLastRows) {
    for (size_t rows : {1u, 2u}) {
        check_host_tile_reference<float>(8, 3, rows, true);
        check_host_tile_reference<ov::float16>(8, 3, rows, true);
        check_host_tile_reference<float>(8, 1, rows, true, 0);
        check_host_tile_reference<ov::float16>(8, 1, rows, true, 0);
    }
}

TEST(HFAHostMaskTest, RegularGapTileMatchesFullReference) {
    check_host_tile_reference<float>(8, 3, 8, false);
    check_host_tile_reference<ov::float16>(8, 3, 2, false);
}

TEST(HFAHostMaskTest, PoolReusesContentAndIsBoundedAcrossChunks) {
    auto row = make_host_row(64, 13);
    auto key = hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row), 8, 8, 8, 8, false, ov::element::f32);
    hfa::HFARuntimeContext ctx;
    size_t allocations = 0;
    auto allocate = [&](const ov::element::Type& dtype, const ov::Shape& shape, const std::string&) {
        ++allocations;
        return ov::get_tensor_impl(ov::Tensor(dtype, shape));
    };
    auto first = ctx.get_host_mask_tile(key, "CPU", allocate);
    ctx.clear_mask_cache();
    EXPECT_EQ(ctx.get_host_mask_tile(key, "CPU", allocate)._ptr, first._ptr);
    EXPECT_EQ(allocations, 1u);
    key.visibility[0] = 0;
    EXPECT_NE(ctx.get_host_mask_tile(key, "CPU", allocate)._ptr, first._ptr);
    EXPECT_EQ(allocations, 2u);
    for (size_t i = 0; i < 16; ++i) {
        for (size_t c = 0; c < 4; ++c) {
            key.visibility[c] = (i >> c) & 1;
        }
        ctx.get_host_mask_tile(key, "CPU", allocate);
        EXPECT_LE(ctx.m_host_mask_pool.size(), ctx.kHostMaskPoolCapacity);
    }
    EXPECT_LE(allocations, ctx.kHostMaskPoolCapacity);
    ctx.reset();
    EXPECT_TRUE(ctx.m_host_mask_pool.empty());
}

TEST(HFAHostMaskTest, FactoryBuildsThreeTilesOnlyForHostMask) {
    for (bool fused : {false, true}) {
        auto model = build_sdpa_model();
        model->get_parameters().back()->set_partial_shape({1, 1, 1, QUERY_SIZE + PAST_LEN});
        model->validate_nodes_and_infer_types();
        annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
        auto off = ov::npuw::function::HostFlashAttention::from(model, fused, true);
        ASSERT_TRUE(off);
        EXPECT_FALSE(off->_host_mask);
        EXPECT_EQ(off->_host_mask_window, 0);
        EXPECT_EQ(off->_masked_tile_model, nullptr);
        for (const auto& node : model->get_ops()) {
            if (node->get_friendly_name() == "add.0") {
                node->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
            }
        }
        auto on = ov::npuw::function::HostFlashAttention::from(model, fused, false);
        ASSERT_TRUE(on);
        EXPECT_TRUE(on->_host_mask);
        EXPECT_EQ(on->_host_mask_window, 0);
        EXPECT_EQ(on->_tile_model->inputs().size(), 6u);
        ASSERT_NE(on->_masked_tile_model, nullptr);
        EXPECT_EQ(on->_masked_tile_model->inputs().size(), 7u);
        EXPECT_EQ(on->_final_tile_model->inputs().size(), 7u);
        for (size_t i = 0; i < 6; ++i) {
            EXPECT_EQ(on->_tile_model->input(i).get_partial_shape(),
                      on->_masked_tile_model->input(i).get_partial_shape());
        }
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(on->_tile_model->output(i).get_partial_shape(),
                      on->_masked_tile_model->output(i).get_partial_shape());
        }
    }
}

TEST(HFAHostMaskTest, FactoryUsesOriginalPrefillLengthForSlicedQuery) {
    for (bool fused : {false, true}) {
        auto model = build_sdpa_model();
        model->get_parameters().front()->set_partial_shape({1, NUM_HEADS, 1, HEAD_DIM});
        model->get_parameters().back()->set_partial_shape({1, 1, 1, QUERY_SIZE + PAST_LEN});
        for (const auto& node : model->get_ops()) {
            if (node->get_friendly_name() == "matmul2.0") {
                node->get_rt_info()[ov::npuw::NPUW_ORIGINAL_QUERY_LENGTH_RT_KEY] = QUERY_SIZE;
            }
        }
        model->validate_nodes_and_infer_types();
        auto hfa = ov::npuw::function::HostFlashAttention::from(model, fused);
        ASSERT_TRUE(hfa);
        EXPECT_EQ(hfa->_query_size, QUERY_SIZE);
        EXPECT_EQ(hfa->_past_tile_size, QUERY_SIZE);
        EXPECT_EQ(hfa->_final_tile_size, QUERY_SIZE);
        EXPECT_EQ(hfa->_tile_model->input(0).get_shape(), (ov::Shape{1, NUM_HEADS, 1, HEAD_DIM}));
        EXPECT_EQ(hfa->_final_tile_model->input(6).get_shape(), (ov::Shape{1, 1, 1, QUERY_SIZE}));
    }
}

TEST(HFAHostMaskTest, FactoryFindsRowThroughReshapeConvert) {
    using namespace ov;
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    auto old = model->get_parameters().back();
    auto row = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, QUERY_SIZE + PAST_LEN});
    auto shape = op::v0::Constant::create<size_t>(element::i64, Shape{4}, {1, 1, 1, QUERY_SIZE + PAST_LEN});
    auto reshape = std::make_shared<op::v1::Reshape>(row, shape, false);
    auto cvt = std::make_shared<op::v0::Convert>(reshape, element::f32);
    for (const auto& node : model->get_ops()) {
        if (node->get_friendly_name() == "add.0") {
            node->input(1).replace_source_output(cvt);
            node->get_rt_info()[npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
        }
    }
    model->remove_parameter(old);
    model->add_parameters({row});
    model->validate_nodes_and_infer_types();
    auto result = npuw::function::HostFlashAttention::from(model, false);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->_attention_mask_param_idx, 5u);
    EXPECT_EQ(result->_final_tile_model->input(6).get_element_type(), element::f32);
}

TEST(HFAHostMaskTest, FactoryPropagatesWindowToCompiledDescriptor) {
    auto model = build_sdpa_model(8, 16, 2, 4);
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (node->get_friendly_name() == "add.0") {
            node->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
            node->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_WINDOW_RT_KEY] = int64_t{25};
        }
    }
    const auto function = ov::npuw::function::HostFlashAttention::from(model, false);
    ASSERT_TRUE(function);
    EXPECT_TRUE(function->_host_mask);
    EXPECT_EQ(function->_host_mask_window, 25);
    const ov::npuw::compiled::HostFlashAttention desc(*function);
    EXPECT_TRUE(desc._host_mask);
    EXPECT_EQ(desc._host_mask_window, 25);
}

TEST(HFAHostMaskTest, FailedConversionThrowsInsteadOfFallingBack) {
    auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 8});
    auto add = std::make_shared<ov::op::v1::Add>(p, p);
    add->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{p});
    try {
        ov::npuw::function::HostFlashAttention::from(model);
        FAIL() << "Host-mask conversion must not fall back";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("NPUW_ATTN_HFA_HOST_MASK"), std::string::npos);
    }
}

TEST(HFAHostMaskTest, PassOffLeavesOriginalProducer) {
    std::shared_ptr<ov::Model> model;
    auto sdpa = make_host_pass_model(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto original = sdpa->input_value(3);
    EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, false));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassOnReplacesCausalAndCloneKeepsAnnotation) {
    std::shared_ptr<ov::Model> original;
    make_host_pass_model(original, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto model = original->clone();
    EXPECT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_TRUE(ov::npuw::has_hfa_host_mask(model));
    for (const auto& node : model->get_ops()) {
        EXPECT_NE(node->get_friendly_name(), "old_full_mask");
        if (auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(sdpa->get_input_node_shared_ptr(3));
            ASSERT_NE(cvt, nullptr);
            auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(cvt->get_input_node_shared_ptr(0));
            ASSERT_NE(reshape, nullptr);
            EXPECT_EQ(reshape->get_shape(), (ov::Shape{1, 1, 1, 24}));
        }
    }
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::pass::ScaledDotProductAttentionDecomposition>();
    rewrite.run_on_model(model);
    EXPECT_TRUE(ov::npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassLeavesSlidingWindowProducer) {
    std::shared_ptr<ov::Model> model;
    auto sdpa = make_host_pass_model(model, 8);
    auto original = sdpa->input_value(3);
    EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, CatalogueF1ExpandedGatherIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F1);
}

TEST(HFAHostMaskTest, CatalogueGemma3nRelativeRowsAreReplaced) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    expect_catalogue_mask_replaced(catalogue::Family::F1, catalogue::Layout::Gemma3n);
}

TEST(HFAHostMaskTest, PropagatedQuerySuffixPreservesMaskEligibility) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (int64_t kind : {npuw::NPUW_SDPA_MASK_CAUSAL, int64_t{8}, int64_t{25}}) {
        SCOPED_TRACE(kind);
        std::shared_ptr<Model> model;
        auto sdpa = make_host_pass_model(model, kind, kind < 0 ? catalogue::Family::F1 : catalogue::Family::F2);
        if (kind < 0) {
            sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
            npuw::DetectAttentionMask().run_on_model(model);
        }
        auto mask_result = std::make_shared<op::v0::Result>(sdpa->input_value(3));
        model->add_results({mask_result});
        auto result = model->get_results().front();
        auto suffix = std::make_shared<op::v8::Slice>(sdpa,
                                                      catalogue::vector({7}),
                                                      catalogue::vector({8}),
                                                      catalogue::vector({1}),
                                                      catalogue::vector({2}));
        result->input(0).replace_source_output(suffix);
        model->validate_nodes_and_infer_types();

        npuw::PropagateSliceUp().run_on_model(model);
        model->remove_result(mask_result);

        auto sliced = as_type_ptr<op::v13::ScaledDotProductAttention>(result->get_input_node_shared_ptr(0));
        ASSERT_NE(sliced, nullptr);
        ASSERT_EQ(sliced->get_input_shape(0), (Shape{1, 2, 1, 4}));
        EXPECT_EQ(npuw::find_propagated_original_query_length(sliced), std::optional<size_t>{8});
        const auto annotation = sliced->get_rt_info().find(npuw::NPUW_SDPA_MASK_RT_KEY);
        ASSERT_NE(annotation, sliced->get_rt_info().end());
        EXPECT_EQ(annotation->second.as<int64_t>(), kind);
        const auto mask = sliced->input_value(3);
        EXPECT_FALSE(npuw::replace_hfa_host_mask(model, false));
        EXPECT_EQ(sliced->input_value(3), mask);
        EXPECT_EQ(npuw::replace_hfa_host_mask(model, true), kind != 8);
        EXPECT_EQ(npuw::has_hfa_host_mask(model), kind != 8);
        if (kind == 8) {
            EXPECT_EQ(sliced->input_value(3), mask);
        }
    }
}

TEST(HFAHostMaskTest, DetectedF1NestedMatrixGatherKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (size_t kind = 0; kind < 3; ++kind) {
        SCOPED_TRACE(kind);
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model,
                                         npuw::NPUW_SDPA_MASK_CAUSAL,
                                         catalogue::Family::F1,
                                         catalogue::Layout::Gemma3n,
                                         false,
                                         &cone);
        ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
        auto row_add = ov::as_type_ptr<op::v1::Add>(cone.rows.get_node_shared_ptr()
                                                        ->get_input_node_shared_ptr(0)
                                                        ->get_input_node_shared_ptr(0)
                                                        ->get_input_node_shared_ptr(0));
        ASSERT_NE(row_add, nullptr);
        Output<Node> matrix;
        if (kind == 0) {
            matrix = op::v0::Constant::create(element::i64, Shape{2, 2}, {0, 16, 15, 0});
        } else if (kind == 1) {
            auto shape_source = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 16, 15, 1});
            model->add_parameters({shape_source});
            matrix = std::make_shared<op::v1::Reshape>(std::make_shared<op::v3::ShapeOf>(shape_source),
                                                       catalogue::vector({2, 2}),
                                                       false);
        } else {
            auto values = catalogue::concat({catalogue::vector({0, 16}), catalogue::vector({15, 0})});
            auto expanded = std::make_shared<op::v1::Reshape>(values, catalogue::vector({1, 2, 2}), false);
            matrix = std::make_shared<op::v0::Squeeze>(expanded, catalogue::scalar(0));
        }
        auto part = std::make_shared<op::v8::Gather>(matrix, catalogue::scalar(1), catalogue::scalar(0), 0);
        auto bound = std::make_shared<op::v8::Gather>(part, catalogue::scalar(0), catalogue::scalar(0), 0);
        row_add->input(1).replace_source_output(bound);
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
        ASSERT_EQ(matrix.get_shape(), (Shape{2, 2}));
        ASSERT_EQ(part->get_shape(), (Shape{2}));
        ASSERT_EQ(bound->get_shape(), Shape{});
        ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
        sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
        npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, DetectedMatchingF1WithCausalAttributeKeepsGraphMask) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
    ASSERT_GT(sdpa->input_value(3).get_shape().back(), sdpa->input_value(0).get_shape()[2]);
    sdpa->set_causal(true);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_TRUE(sdpa->get_causal());
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueF1NoncanonicalRangeStartAndStepKeepGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (bool nonunit_step : {false, true}) {
        SCOPED_TRACE(nonunit_step);
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model,
                                         npuw::NPUW_SDPA_MASK_CAUSAL,
                                         catalogue::Family::F1,
                                         catalogue::Layout::Gemma3n,
                                         false,
                                         &cone);
        ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
        auto row_add = cone.rows.get_node_shared_ptr()
                           ->get_input_node_shared_ptr(0)
                           ->get_input_node_shared_ptr(0)
                           ->get_input_node_shared_ptr(0);
        auto range = ov::as_type_ptr<op::v4::Range>(row_add->get_input_node_shared_ptr(0));
        ASSERT_NE(range, nullptr);
        range->input(0).replace_source_output(catalogue::scalar(nonunit_step ? 0 : 1, element::i32));
        range->input(1).replace_source_output(catalogue::scalar(nonunit_step ? 16 : 9));
        range->input(2).replace_source_output(catalogue::scalar(nonunit_step ? 2 : 1, element::i32));
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
        ASSERT_EQ(range->get_shape(), (Shape{8}));
        ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CatalogueF1SwappedRowColumnComparisonKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    catalogue::Mask cone;
    auto sdpa = make_host_pass_model(model,
                                     npuw::NPUW_SDPA_MASK_CAUSAL,
                                     catalogue::Family::F1,
                                     catalogue::Layout::Direct,
                                     false,
                                     &cone);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
    auto compare =
        ov::as_type_ptr<op::v1::LessEqual>(cone.predicates->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(1));
    ASSERT_NE(compare, nullptr);
    compare->input(0).replace_source_output(cone.rows);
    compare->input(1).replace_source_output(cone.columns);
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueF1NumpyBroadcastKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    catalogue::Mask cone;
    auto sdpa = make_host_pass_model(model,
                                     npuw::NPUW_SDPA_MASK_CAUSAL,
                                     catalogue::Family::F1,
                                     catalogue::Layout::Direct,
                                     false,
                                     &cone);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
    auto broadcast = ov::as_type_ptr<op::v3::Broadcast>(cone.select->get_input_node_shared_ptr(0));
    ASSERT_NE(broadcast, nullptr);
    cone.select->input(0).replace_source_output(std::make_shared<op::v3::Broadcast>(broadcast->input_value(0),
                                                                                    broadcast->input_value(1),
                                                                                    op::BroadcastType::NUMPY));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueF1PaddingGatherAxisOneKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    catalogue::Mask cone;
    auto sdpa = make_host_pass_model(model,
                                     npuw::NPUW_SDPA_MASK_CAUSAL,
                                     catalogue::Family::F1,
                                     catalogue::Layout::Direct,
                                     false,
                                     &cone);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
    auto gather = ov::as_type_ptr<op::v8::Gather>(
        cone.predicates->get_input_node_shared_ptr(1)->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0));
    ASSERT_NE(gather, nullptr);
    auto flat = ov::as_type_ptr<op::v1::Reshape>(gather->get_input_node_shared_ptr(0));
    ASSERT_NE(flat, nullptr);
    flat->input(1).replace_source_output(catalogue::vector({1, 24}));
    gather->input(2).replace_source_output(catalogue::scalar(1, element::i32));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(flat->get_shape(), (Shape{1, 24}));
    ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueF1NaNAndPositiveInfinityKeepGraphMask) {
    using namespace ov;
    for (float masked : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        SCOPED_TRACE(masked);
        std::shared_ptr<Model> model;
        auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
        ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
        sdpa->get_input_node_shared_ptr(3)->input(2).replace_source_output(
            op::v0::Constant::create(element::f32, Shape{}, {masked}));
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, DetectedCatalogueF1F16IsReplaced) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    for (size_t idx : {0u, 1u, 2u}) {
        model->get_parameters()[idx]->set_element_type(element::f16);
    }
    auto select = sdpa->get_input_node_shared_ptr(3);
    select->input(1).replace_source_output(op::v0::Constant::create(element::f16, Shape{}, {0}));
    select->input(2).replace_source_output(op::v0::Constant::create(element::f16, Shape{}, {-65504}));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(sdpa->input_value(3).get_element_type(), element::f16);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_TRUE(npuw::has_hfa_host_mask(model));
    EXPECT_EQ(sdpa->input_value(3).get_element_type(), element::f16);
    EXPECT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 1, 24}));
}

TEST(HFAHostMaskTest, CatalogueF2PatchedWindowIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F2);
}

TEST(HFAHostMaskTest, CatalogueF2RankThreePositionIdsKeepGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    catalogue::Mask cone;
    auto sdpa = make_host_pass_model(model, 25, catalogue::Family::F2, catalogue::Layout::Direct, false, &cone);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
    auto positions = model->get_parameters().back();
    positions->set_partial_shape({1, 1, 8});
    ASSERT_NO_THROW(positions->validate_and_infer_types());
    auto squeezed = std::make_shared<op::v0::Squeeze>(positions, catalogue::scalar(0));
    auto position_offset = cone.position_bound->get_input_node_shared_ptr(1);
    position_offset->input(0).replace_source_output(catalogue::unsqueeze(catalogue::unsqueeze(squeezed, 0), 3));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(positions->get_shape(), (Shape{1, 1, 8}));
    ASSERT_EQ(sdpa->input_value(3).get_shape(), (Shape{1, 1, 8, 24}));
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueF3PaddedGatherIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F3);
}

TEST(HFAHostMaskTest, CatalogueF4aDoubleMaskedFillIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F4a);
}

TEST(HFAHostMaskTest, CatalogueF4bPhiVisionLowerBandIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F4b,
                                   ov::test::npuw::host_mask_catalogue::Layout::Direct,
                                   true);
}

TEST(HFAHostMaskTest, CatalogueF5AdditiveTriuIsReplaced) {
    expect_catalogue_mask_replaced(ov::test::npuw::host_mask_catalogue::Family::F5);
}

TEST(HFAHostMaskTest, CatalogueF5NegativeInfinityArithmeticKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (bool causal_infinity : {false, true}) {
        SCOPED_TRACE(causal_infinity);
        std::shared_ptr<Model> model;
        auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL, catalogue::Family::F5);
        ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
        auto sum = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
        ASSERT_TRUE(ov::is_type<op::v1::Add>(sum));
        auto padding = ov::as_type_ptr<op::v1::Multiply>(sum->get_input_node_shared_ptr(1));
        ASSERT_NE(padding, nullptr);
        padding->input(1).replace_source_output(
            op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {-std::numeric_limits<float>::infinity()}));
        if (causal_infinity) {
            auto causal = ov::as_type_ptr<op::v1::Select>(
                sum->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0));
            ASSERT_NE(causal, nullptr);
            causal->get_input_node_shared_ptr(1)->input(0).replace_source_output(
                op::v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
        }
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CataloguePhi4BothExportsAreReplaced) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (auto layout : {catalogue::Layout::Phi4, catalogue::Layout::Phi4RC2}) {
        SCOPED_TRACE(static_cast<int>(layout));
        expect_catalogue_mask_replaced(catalogue::Family::F2, layout);
    }
}

TEST(HFAHostMaskTest, CatalogueContextSliceAndQuerySuffixAreReplaced) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (auto layout : {catalogue::Layout::ContextSlice, catalogue::Layout::QuerySuffix}) {
        SCOPED_TRACE(static_cast<int>(layout));
        expect_catalogue_mask_replaced(catalogue::Family::F1, layout);
    }
}

TEST(HFAHostMaskTest, CatalogueImplicitAxesSliceKeepsGraphMask) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<ov::Model> model;
    auto sdpa = make_host_pass_model(model,
                                     ov::npuw::NPUW_SDPA_MASK_CAUSAL,
                                     catalogue::Family::F1,
                                     catalogue::Layout::QuerySuffix);
    auto suffix = sdpa->get_input_node_shared_ptr(3);
    auto implicit = std::make_shared<ov::op::v8::Slice>(suffix->input_value(0),
                                                        catalogue::vector({0}),
                                                        catalogue::vector({1}),
                                                        catalogue::vector({1}));
    suffix->input(0).replace_source_output(implicit);
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueUnknownAnnotationKeepsGraphMaskAndCausalAttribute) {
    for (bool causal : {false, true}) {
        std::shared_ptr<ov::Model> model;
        auto sdpa = make_host_pass_model(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
        sdpa->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        sdpa->set_causal(causal);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CatalogueExternalizedSwaKeepsGraphMaskAndCausalAttribute) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_host_pass_model(model, 8);
    auto external = ov::test::utils::make_param(element::f32, Shape{1, 1, 8, 24}, "sliding_window_attention_mask");
    model->add_parameters({external});
    sdpa->input(3).replace_source_output(external);
    sdpa->set_causal(true);
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueTokenTypeMaskKeepsGraphMaskAndCausalAttribute) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa =
        make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL, ov::test::npuw::host_mask_catalogue::Family::F5);
    auto types = ov::test::utils::make_param(element::i64, Shape{1, 24}, "token_type_ids");
    model->add_parameters({types});
    auto sequence = std::make_shared<op::v1::Reshape>(model->get_parameters()[0],
                                                      ov::test::npuw::host_mask_catalogue::vector({1, 8, 8}),
                                                      false);
    sdpa->input(3).replace_source_output(
        ov::test::npuw::make_vlm_bidirectional_modifier(sdpa->input_value(3), types, sequence, element::f32));
    sdpa->set_causal(true);
    model->validate_nodes_and_infer_types();
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CatalogueNarrowWindowKeepsGraphMaskAndCausalAttribute) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (int64_t window : {1, 23}) {
        std::shared_ptr<ov::Model> model;
        auto sdpa = make_host_pass_model(model, window, catalogue::Family::F2);
        sdpa->set_causal(true);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CatalogueUnknownRootKeepsGraphMaskAndCausalAttribute) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    auto unknown = std::make_shared<op::v1::Maximum>(sdpa->input_value(3),
                                                     op::v0::Constant::create(element::f32, Shape{}, {-65504}));
    sdpa->input(3).replace_source_output(unknown);
    sdpa->set_causal(true);
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, CataloguePhiVisionNonredundantBandKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    catalogue::Mask cone;
    auto sdpa = make_host_pass_model(model,
                                     npuw::NPUW_SDPA_MASK_CAUSAL,
                                     catalogue::Family::F4b,
                                     catalogue::Layout::Direct,
                                     true,
                                     &cone);
    cone.lower_start->input(1).replace_source_output(catalogue::scalar(-1));
    expect_host_mask_rejected(model, sdpa);
}

TEST(HFAHostMaskTest, DetectedWindowWithUnrelatedGreaterCutoffKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (int64_t window : {24, 25, 26}) {
        SCOPED_TRACE(window);
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model, window, catalogue::Family::F2, catalogue::Layout::Phi4, false, &cone);
        detect_catalogue_window(model, sdpa, cone);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        EXPECT_EQ(npuw::replace_hfa_host_mask(model->clone(), true), window >= 25);
        auto cutoff = std::make_shared<op::v1::Greater>(cone.columns, catalogue::scalar(5));
        cone.predicates->input(0).replace_source_output(
            std::make_shared<op::v13::BitwiseAnd>(cone.predicates->input_value(0), cutoff));
        detect_catalogue_window(model, sdpa, cone, cutoff);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, DetectedClampedColumnBoundsKeepGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (bool lower : {false, true}) {
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model,
                                         npuw::NPUW_SDPA_MASK_CAUSAL,
                                         catalogue::Family::F1,
                                         catalogue::Layout::Direct,
                                         false,
                                         &cone);
        Output<Node> restriction;
        if (lower) {
            auto bound = std::make_shared<op::v1::Maximum>(catalogue::scalar(1), catalogue::scalar(0));
            restriction = std::make_shared<op::v1::Less>(bound, cone.columns);
        } else {
            auto bound = std::make_shared<op::v1::Minimum>(catalogue::scalar(23), cone.context);
            restriction = std::make_shared<op::v1::Less>(cone.columns, bound);
        }
        cone.predicates->input(0).replace_source_output(
            std::make_shared<op::v13::BitwiseAnd>(cone.predicates->input_value(0), restriction));
        sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
        npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CatalogueWindowRequiresBothSameWindowBoundsAndExactOr) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (size_t kind = 0; kind < 6; ++kind) {
        SCOPED_TRACE(kind);
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model, 25, catalogue::Family::F2, catalogue::Layout::Phi4, false, &cone);
        ASSERT_TRUE(npuw::replace_hfa_host_mask(model->clone(), true));
        if (kind == 0) {
            cone.position_bound->input(1).replace_source_output(
                std::make_shared<op::v1::Add>(cone.rows, catalogue::scalar(-25)));
        } else if (kind == 1) {
            cone.row_bound->get_input_node_shared_ptr(1)->input(1).replace_source_output(
                op::v0::Constant::create(element::i64, Shape{1, 1, 1, 1}, {-26}));
        } else if (kind == 2) {
            cone.window_or->input(1).replace_source_output(
                std::make_shared<op::v1::Less>(cone.columns, catalogue::scalar(24)));
        } else if (kind == 3) {
            cone.window_or->input(0).replace_source_output(op::v0::Constant::create(element::boolean, Shape{}, {true}));
        } else if (kind == 4) {
            sdpa->get_rt_info()[npuw::NPUW_SDPA_MASK_RT_KEY] = int64_t{26};
        } else {
            auto predicate = cone.predicates->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(1);
            predicate->input(1).replace_source_output(op::v0::Constant::create(element::boolean, Shape{}, {true}));
        }
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, CatalogueGatherRequiresWidthSplitOutput) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (auto family : {catalogue::Family::F1, catalogue::Family::F3}) {
        std::shared_ptr<Model> model;
        catalogue::Mask cone;
        auto sdpa =
            make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL, family, catalogue::Layout::Direct, false, &cone);
        const auto targets = cone.index_split->output(1).get_target_inputs();
        for (const auto& input : targets) {
            input.replace_source_output(cone.index_split->output(0));
        }
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, SyntheticTriuWindowAlwaysKeepsGraphMask) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (int64_t window : {23, 24, 25}) {
        std::shared_ptr<Model> model;
        auto sdpa = make_host_pass_model(model, window);
        auto sequence =
            std::make_shared<op::v1::Reshape>(model->get_parameters()[0], catalogue::vector({1, 8, 8}), false);
        auto coordinates = ov::test::npuw::make_causal_bool(sequence, model->get_parameters()[3], "triu.");
        auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
        auto lowest = op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});
        auto causal = std::make_shared<op::v1::Select>(
            std::make_shared<op::v1::GreaterEqual>(coordinates.q_col, coordinates.kv_row),
            zero,
            lowest);
        auto band = std::make_shared<op::v1::GreaterEqual>(
            std::make_shared<op::v1::Subtract>(coordinates.q_col, coordinates.kv_row),
            catalogue::scalar(window));
        auto window_mask = std::make_shared<op::v1::Select>(band, lowest, causal);
        auto row_bool = std::make_shared<op::v0::Convert>(model->get_parameters()[3], element::boolean);
        auto padding = std::make_shared<op::v1::Select>(row_bool, zero, lowest);
        sdpa->input(3).replace_source_output(
            catalogue::unsqueeze(catalogue::unsqueeze(std::make_shared<op::v1::Add>(window_mask, padding), 0), 1));
        sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
        npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        expect_host_mask_rejected(model, sdpa);
    }
}

TEST(HFAHostMaskTest, DescriptorSerializationRoundTrip) {
    for (bool host : {false, true}) {
        ov::npuw::compiled::HostFlashAttention desc;
        desc._past_tile_size = 8;
        desc._final_tile_size = 8;
        desc._sdpa_attention_info._query_size = 8;
        desc._host_mask = host;
        desc._host_mask_window = host ? 25 : 0;
        std::stringstream blob;
        auto writer = ov::npuw::orc::Stream::writer(blob);
        writer & desc;
        ov::npuw::compiled::HostFlashAttention restored;
        auto reader = ov::npuw::orc::Stream::reader(blob);
        reader & restored;
        EXPECT_EQ(restored._host_mask, host);
        EXPECT_EQ(restored._host_mask_window, host ? 25 : 0);
        EXPECT_EQ(restored._past_tile_size, 8);
        EXPECT_EQ(restored._host_mask_kind, ov::npuw::HFAHostMaskKind::Causal);
    }
}

TEST(HFAHostMaskTest, OffDescriptorKeepsLegacyWireLayout) {
    ov::npuw::compiled::HostFlashAttention desc;
    desc._past_tile_size = 16;
    desc._final_tile_size = 16;
    desc._sdpa_attention_info._query_size = 16;
    desc._sdpa_attention_info._context_size = 64;
    desc._host_mask_window = 25;
    std::stringstream actual, legacy;
    auto writer = ov::npuw::orc::Stream::writer(actual);
    writer & desc;
    auto old_writer = ov::npuw::orc::Stream::writer(legacy);
    auto& info = desc._sdpa_attention_info;
    old_writer & info._query_size & info._context_size & info._k_seq_dim & info._v_seq_dim & info._sdpa_indices.query &
        info._sdpa_indices.past_key_blocks & info._sdpa_indices.past_value_blocks & info._sdpa_indices.present_key &
        info._sdpa_indices.present_value & info._sdpa_indices.attention_mask & info._tile_input_indices.q &
        info._tile_input_indices.k & info._tile_input_indices.v & info._tile_input_indices.mask &
        info._tile_input_indices.acc & info._tile_input_indices.max & info._tile_input_indices.d &
        info._tile_output_indices.acc & info._tile_output_indices.max & info._tile_output_indices.d &
        desc._past_tile_size & desc._final_tile_size & desc._can_use_tensor_view;
    EXPECT_EQ(actual.str(), legacy.str());
    ov::npuw::compiled::HostFlashAttention restored;
    auto reader = ov::npuw::orc::Stream::reader(actual);
    reader & restored;
    EXPECT_FALSE(restored._host_mask);
    EXPECT_EQ(restored._host_mask_window, 0);
}

TEST(HFAHostMaskTest, OnDescriptorRejectsVersionOneExtension) {
    ov::npuw::compiled::HostFlashAttention desc;
    desc._past_tile_size = 8;
    desc._final_tile_size = 8;
    desc._sdpa_attention_info._query_size = 8;
    std::stringstream off_blob;
    auto off_writer = ov::npuw::orc::Stream::writer(off_blob);
    off_writer & desc;
    const auto extension_offset = off_blob.str().size();
    desc._host_mask = true;
    desc._host_mask_window = 25;
    std::stringstream on_blob;
    auto on_writer = ov::npuw::orc::Stream::writer(on_blob);
    on_writer & desc;
    auto version_one = on_blob.str();
    ASSERT_GT(version_one.size(), extension_offset + 2);
    ASSERT_EQ(static_cast<uint8_t>(version_one[extension_offset]), 2);
    version_one[extension_offset] = 1;
    version_one.resize(extension_offset + 2);
    std::stringstream legacy(version_one);
    auto reader = ov::npuw::orc::Stream::reader(legacy);
    ov::npuw::compiled::HostFlashAttention restored;
    try {
        reader & restored;
        FAIL() << "The version 1 ON extension has no runtime window bound";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Unsupported HFA host-mask extension"), std::string::npos);
    }
}

TEST(HFAHostMaskTest, HostModeDoesNotAllocateEagerMaskBuffers) {
    auto model = build_sdpa_model();
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (node->get_friendly_name() == "add.0") {
            node->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
        }
    }
    auto function = ov::npuw::function::HostFlashAttention::from(model, false);
    ASSERT_TRUE(function);
    ov::npuw::compiled::HostFlashAttention desc(*function);
    auto plugin = std::make_shared<PosIdsNullPlugin>();
    desc._compiled_tile_model = {std::make_shared<PosIdsStubCompiledModel>(function->_tile_model, plugin), {}};
    desc._compiled_masked_tile_model = {std::make_shared<PosIdsStubCompiledModel>(function->_masked_tile_model, plugin),
                                        {}};
    desc._compiled_final_tile_model = {std::make_shared<PosIdsStubCompiledModel>(function->_final_tile_model, plugin),
                                       {}};
    hfa::HFARuntimeContext ctx;
    size_t allocations = 0;
    ctx.initialize_mask_cache(desc,
                              "CPU",
                              [&](const ov::element::Type& dtype, const ov::Shape& shape, const std::string&) {
                                  ++allocations;
                                  return ov::get_tensor_impl(ov::Tensor(dtype, shape));
                              });
    EXPECT_TRUE(desc.is_valid());
    EXPECT_EQ(allocations, 0u);
    EXPECT_TRUE(ctx.m_mask_tile_buffers.empty());
    EXPECT_FALSE(ctx.m_final_mask_tile_buffer);
}

TEST(HFAHostMaskTest, PassSharesRowAcrossCausalLayersAndRejectsBooleanMask) {
    std::shared_ptr<ov::Model> model;
    auto first = make_host_pass_model(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto second = first->clone_with_new_inputs(first->input_values());
    second->get_rt_info() = first->get_rt_info();
    const auto graph_mask = second->get_input_node_shared_ptr(3);
    model->add_results({std::make_shared<ov::op::v0::Result>(second)});
    ASSERT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(first->input_value(3), second->input_value(3));
    auto row = model->get_parameters().back();
    auto bool_mask = std::make_shared<ov::op::v0::Convert>(row, ov::element::boolean);
    first->input(3).replace_source_output(bool_mask);
    first->get_rt_info().erase(ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY);
    second->input(3).replace_source_output(graph_mask);
    second->get_rt_info().erase(ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY);
    EXPECT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(first->get_input_node_shared_ptr(3), bool_mask);
    EXPECT_EQ(first->get_rt_info().count(ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY), 0u);
}

TEST(HFAHostMaskTest, PassReplacesWindowsCoveringContext) {
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    for (int64_t window : {23, 24, 25, 26, 262144}) {
        std::shared_ptr<ov::Model> model;
        catalogue::Mask cone;
        auto sdpa = make_host_pass_model(model, window, catalogue::Family::F2, catalogue::Layout::Phi4, false, &cone);
        detect_catalogue_window(model, sdpa, cone);
        ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        if (window < 25) {
            expect_host_mask_rejected(model, sdpa);
        } else {
            EXPECT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
            EXPECT_TRUE(ov::npuw::has_hfa_host_mask(model));
            EXPECT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_HFA_HOST_MASK_WINDOW_RT_KEY).as<int64_t>(), window);
        }
    }
}

TEST(HFAHostMaskTest, PassStampsMinimumReplacedWindowAcrossMixedF1F2Layers) {
    using namespace ov;
    namespace catalogue = ov::test::npuw::host_mask_catalogue;
    std::shared_ptr<Model> model;
    auto first = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    auto positions = ov::test::utils::make_param(element::i64, Shape{1, 8}, "position_ids");
    model->add_parameters({positions});
    const auto& parameters = model->get_parameters();
    auto sequence = std::make_shared<op::v1::Reshape>(parameters[0], catalogue::vector({1, 8, 8}), false);
    std::vector<std::shared_ptr<op::v13::ScaledDotProductAttention>> layers{first};
    for (int64_t window : {26, 25, 24}) {
        auto mask =
            catalogue::make_mask(catalogue::Family::F2, parameters[3], parameters[1], sequence, positions, window);
        auto sdpa = std::make_shared<op::v13::ScaledDotProductAttention>(first->input_value(0),
                                                                         first->input_value(1),
                                                                         first->input_value(2),
                                                                         mask.root,
                                                                         false);
        sdpa->get_rt_info()[npuw::NPUW_SDPA_MASK_RT_KEY] = window;
        model->add_results({std::make_shared<op::v0::Result>(sdpa)});
        layers.push_back(sdpa);
    }
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    const auto rejected_mask = layers.back()->input_value(3);
    ASSERT_TRUE(npuw::replace_hfa_host_mask(model, true));
    for (size_t idx = 0; idx < 3; ++idx) {
        SCOPED_TRACE(idx);
        EXPECT_TRUE(layers[idx]->get_rt_info().at(npuw::NPUW_HFA_HOST_MASK_RT_KEY).as<bool>());
        EXPECT_EQ(layers[idx]->get_rt_info().at(npuw::NPUW_HFA_HOST_MASK_WINDOW_RT_KEY).as<int64_t>(), 25);
        EXPECT_EQ(layers[idx]->input_value(3), first->input_value(3));
    }
    EXPECT_EQ(layers.back()->input_value(3), rejected_mask);
    EXPECT_FALSE(layers.back()->get_causal());
    EXPECT_EQ(layers.back()->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), 24);
    EXPECT_EQ(layers.back()->get_rt_info().count(npuw::NPUW_HFA_HOST_MASK_RT_KEY), 0u);
    EXPECT_EQ(layers.back()->get_rt_info().count(npuw::NPUW_HFA_HOST_MASK_WINDOW_RT_KEY), 0u);
}

TEST(HFAHostMaskTest, FactoryAcceptsWindowsCoveringContextOnly) {
    constexpr int64_t context = QUERY_SIZE + PAST_LEN;
    for (int64_t window : {context - 1, context, context + 1}) {
        auto model = build_sdpa_model();
        annotate_mask_rt_info(model, window);
        for (const auto& node : model->get_ops()) {
            if (node->get_friendly_name() == "add.0") {
                node->get_rt_info()[ov::npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
            }
        }
        if (window < context) {
            EXPECT_THROW(ov::npuw::function::HostFlashAttention::from(model, false), ov::Exception);
        } else {
            auto hfa = ov::npuw::function::HostFlashAttention::from(model, false);
            ASSERT_TRUE(hfa);
            EXPECT_TRUE(hfa->_host_mask);
            EXPECT_NE(hfa->_masked_tile_model, nullptr);
        }
    }
}

TEST(HFAHostMaskTest, DetectedImageBlockMaskIsNotReplaced) {
    std::shared_ptr<ov::Model> model;
    auto sdpa = make_detected_host_pass_model(model, true);
    ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, DetectedCausalMaskAllowsShapeOnlyInputs) {
    for (bool embeds : {false, true}) {
        std::shared_ptr<ov::Model> model;
        auto sdpa = make_host_pass_model(model,
                                         ov::npuw::NPUW_SDPA_MASK_CAUSAL,
                                         ov::test::npuw::host_mask_catalogue::Family::F1,
                                         ov::test::npuw::host_mask_catalogue::Layout::Direct,
                                         embeds);
        sdpa->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        ov::npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(),
                  ov::npuw::NPUW_SDPA_MASK_CAUSAL);
        EXPECT_TRUE(ov::npuw::replace_hfa_host_mask(model, true));
        EXPECT_TRUE(ov::npuw::has_hfa_host_mask(model));
    }
}

TEST(HFAHostMaskTest, PassRejectsOrXorEvenWithoutExtraParameters) {
    using namespace ov;
    for (size_t kind = 0; kind < 4; ++kind) {
        std::shared_ptr<Model> model;
        auto sdpa = make_detected_host_pass_model(model, false);
        auto mask = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
        auto visible = mask->input_value(0);
        auto identity = op::v0::Constant::create(element::boolean, Shape{}, {false});
        std::shared_ptr<Node> combined;
        switch (kind) {
        case 0:
            combined = std::make_shared<op::v13::BitwiseOr>(visible, identity);
            break;
        case 1:
            combined = std::make_shared<op::v1::LogicalOr>(visible, identity);
            break;
        case 2:
            combined = std::make_shared<op::v1::LogicalXor>(visible, identity);
            break;
        case 3:
            combined = std::make_shared<op::v13::BitwiseXor>(visible, identity);
            break;
        }
        mask->input(0).replace_source_output(combined);
        const auto original = sdpa->input_value(3);
        EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
        EXPECT_EQ(sdpa->input_value(3), original);
    }
}

TEST(HFAHostMaskTest, PassRejectsAdditiveValueParametersIncludingPositionIds) {
    using namespace ov;
    for (const auto& name : {"token_type_ids", "cache_position", "beam_idx", "position_ids"}) {
        std::shared_ptr<Model> model;
        auto sdpa = make_detected_host_pass_model(model, false);
        auto extra = ov::test::utils::make_param(element::i64, Shape{1, 24}, name);
        model->add_parameters({extra});
        auto mask =
            std::make_shared<op::v1::Add>(sdpa->input_value(3), std::make_shared<op::v0::Convert>(extra, element::f32));
        sdpa->input(3).replace_source_output(mask);
        sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
        npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
        EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
        EXPECT_FALSE(npuw::has_hfa_host_mask(model));
        EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    }
}

TEST(HFAHostMaskTest, PassRejectsPerColumnBiasEvenWhenSharedWithPredicate) {
    using namespace ov;
    for (bool shared : {false, true}) {
        std::shared_ptr<Model> model;
        auto sdpa = make_detected_host_pass_model(model, false);
        std::vector<float> values(24, 0);
        values.back() = 1;
        auto bias = op::v0::Constant::create(element::f32, Shape{1, 1, 1, 24}, values);
        Output<Node> base = sdpa->input_value(3);
        if (shared) {
            auto condition =
                std::make_shared<op::v1::Equal>(bias, op::v0::Constant::create(element::f32, Shape{}, {0}));
            base = std::make_shared<op::v1::Select>(condition, base, base);
        }
        auto mask = std::make_shared<op::v1::Add>(bias, base);
        sdpa->input(3).replace_source_output(mask);
        sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
        npuw::DetectAttentionMask().run_on_model(model);
        ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
        EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
        EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    }
}

TEST(HFAHostMaskTest, PassRejectsConstantColumnRestriction) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
    std::vector<char> columns(24, true);
    columns.back() = false;
    auto restriction = op::v0::Constant::create(element::boolean, Shape{1, 24}, columns);
    select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), restriction));
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsSelectDerivedColumnBias) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    std::vector<char> columns(24, false);
    columns.back() = true;
    auto condition = op::v0::Constant::create(element::boolean, Shape{1, 24}, columns);
    auto bias = std::make_shared<op::v1::Select>(condition,
                                                 op::v0::Constant::create(element::f32, Shape{}, {1}),
                                                 op::v0::Constant::create(element::f32, Shape{}, {0}));
    auto mask = std::make_shared<op::v1::Add>(sdpa->input_value(3), bias);
    sdpa->input(3).replace_source_output(mask);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsUniformFiniteBias) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto mask =
        std::make_shared<op::v1::Add>(sdpa->input_value(3), op::v0::Constant::create(element::f32, Shape{}, {0.5f}));
    sdpa->input(3).replace_source_output(mask);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsScalarColumnCutoff) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
    auto columns = std::make_shared<op::v4::Range>(op::v0::Constant::create(element::i64, Shape{}, {0}),
                                                   op::v0::Constant::create(element::i64, Shape{}, {24}),
                                                   op::v0::Constant::create(element::i64, Shape{}, {1}),
                                                   element::i64);
    auto restriction = std::make_shared<op::v1::Less>(columns, op::v0::Constant::create(element::i64, Shape{}, {22}));
    select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), restriction));
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsPositionIdsComparedWithConstant) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto position_ids = ov::test::utils::make_param(element::i64, Shape{1, 8}, "position_ids");
    model->add_parameters({position_ids});
    auto column = std::make_shared<op::v1::Reshape>(position_ids,
                                                    op::v0::Constant::create(element::i64, Shape{2}, {8, 1}),
                                                    false);
    auto nonnegative =
        std::make_shared<op::v1::GreaterEqual>(column, op::v0::Constant::create(element::i64, Shape{}, {0}));
    auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
    select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), nonnegative));
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsMRopePositionIdsComparedWithConstant) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto positions = ov::test::utils::make_param(element::i64, Shape{3, 1, 8}, "position_ids");
    model->add_parameters({positions});
    auto section = std::make_shared<op::v8::Gather>(positions,
                                                    op::v0::Constant::create(element::i64, Shape{}, {0}),
                                                    op::v0::Constant::create(element::i64, Shape{}, {0}));
    auto column =
        std::make_shared<op::v1::Reshape>(section, op::v0::Constant::create(element::i64, Shape{2}, {-1, 1}), false);
    auto nonnegative =
        std::make_shared<op::v1::GreaterEqual>(column, op::v0::Constant::create(element::i64, Shape{}, {0}));
    auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
    select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), nonnegative));
    model->validate_nodes_and_infer_types();
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsUncataloguedArithmeticPaddingValueSet) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto row = std::make_shared<op::v0::Convert>(model->get_parameters()[3], element::f32);
    auto inverted = std::make_shared<op::v1::Subtract>(op::v0::Constant::create(element::f32, Shape{}, {1}), row);
    auto padding = std::make_shared<op::v1::Multiply>(
        inverted,
        op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()}));
    sdpa->input(3).replace_source_output(std::make_shared<op::v1::Add>(sdpa->input_value(3), padding));
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassAllowsSelectMaskValueSet) {
    using namespace ov;
    for (float masked : {std::numeric_limits<float>::lowest(), -std::numeric_limits<float>::infinity()}) {
        std::shared_ptr<Model> model;
        auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
        auto select = sdpa->get_input_node_shared_ptr(3);
        select->input(2).replace_source_output(op::v0::Constant::create(element::f32, Shape{}, {masked}));
        EXPECT_TRUE(npuw::replace_hfa_host_mask(model, true));
        EXPECT_TRUE(npuw::has_hfa_host_mask(model));
    }
}

TEST(HFAHostMaskTest, PassRejectsUncataloguedSumOfCausalAndPaddingMasks) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto causal = ov::test::npuw::make_causal_bool(model->get_parameters()[4], model->get_parameters()[3], "sum.");
    auto zero = op::v0::Constant::create(element::f32, Shape{}, {0});
    auto lowest = op::v0::Constant::create(element::f32, Shape{}, {std::numeric_limits<float>::lowest()});
    auto causal_mask = std::make_shared<op::v1::Select>(causal.mask, zero, lowest);
    auto row = std::make_shared<op::v0::Convert>(model->get_parameters()[3], element::boolean);
    auto padding_mask = std::make_shared<op::v1::Select>(row, zero, lowest);
    sdpa->input(3).replace_source_output(std::make_shared<op::v1::Add>(causal_mask, padding_mask));
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsUncataloguedPaddingComparedWithConstant) {
    using namespace ov;
    for (bool greater : {false, true}) {
        std::shared_ptr<Model> model;
        auto sdpa = make_detected_host_pass_model(model, false);
        auto row = model->get_parameters()[3];
        auto zero = op::v0::Constant::create(element::i64, Shape{}, {0});
        std::shared_ptr<Node> padding;
        if (greater) {
            padding = std::make_shared<op::v1::Greater>(row, zero);
        } else {
            padding = std::make_shared<op::v1::NotEqual>(row, zero);
        }
        auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
        select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), padding));
        const auto original = sdpa->input_value(3);
        EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
        EXPECT_EQ(sdpa->input_value(3), original);
        EXPECT_FALSE(npuw::has_hfa_host_mask(model));
    }
}

TEST(HFAHostMaskTest, PassRejectsSubtractedFiniteBias) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto mask = std::make_shared<op::v1::Subtract>(sdpa->input_value(3),
                                                   op::v0::Constant::create(element::f32, Shape{}, {0.5f}));
    sdpa->input(3).replace_source_output(mask);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsAttenuatedMask) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto mask = std::make_shared<op::v1::Multiply>(sdpa->input_value(3),
                                                   op::v0::Constant::create(element::f32, Shape{}, {0.5f}));
    sdpa->input(3).replace_source_output(mask);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->get_input_node_shared_ptr(3), mask);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, PassRejectsUncataloguedPositionIdsComparedWithCoordinates) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_detected_host_pass_model(model, false);
    auto positions = ov::test::utils::make_param(element::i64, Shape{8, 1}, "position_ids");
    model->add_parameters({positions});
    auto coordinates =
        ov::test::npuw::make_causal_bool(model->get_parameters()[4], model->get_parameters()[3], "positions.");
    auto causal = std::make_shared<op::v1::GreaterEqual>(positions, coordinates.kv_row);
    auto select = sdpa->get_input_node_shared_ptr(3)->get_input_node_shared_ptr(0);
    select->input(0).replace_source_output(std::make_shared<op::v13::BitwiseAnd>(select->input_value(0), causal));
    const auto original = sdpa->input_value(3);
    EXPECT_FALSE(npuw::replace_hfa_host_mask(model, true));
    EXPECT_EQ(sdpa->input_value(3), original);
    EXPECT_FALSE(npuw::has_hfa_host_mask(model));
}

TEST(HFAHostMaskTest, StandardCausalBooleanPaddingIsReplaced) {
    using namespace ov;
    std::shared_ptr<Model> model;
    auto sdpa = make_host_pass_model(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    sdpa->get_rt_info().erase(npuw::NPUW_SDPA_MASK_RT_KEY);
    npuw::DetectAttentionMask().run_on_model(model);
    ASSERT_EQ(sdpa->get_rt_info().at(npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), npuw::NPUW_SDPA_MASK_CAUSAL);
    EXPECT_TRUE(npuw::replace_hfa_host_mask(model, true));
}

TEST(HFAHostMaskTest, FinalTileUnmasksOnlyRowsWithoutPastOrPresent) {
    constexpr size_t query = 8, context = 24;
    for (auto dtype : {ov::element::f16, ov::element::f32}) {
        const float lowest = dtype == ov::element::f16 ? static_cast<float>(std::numeric_limits<ov::float16>::lowest())
                                                       : std::numeric_limits<float>::lowest();
        for (size_t n : {1u, 3u}) {
            for (size_t past : {0u, 16u}) {
                for (bool visible_past : {false, true}) {
                    auto row = make_host_row(context, visible_past ? 1 : 0);
                    std::fill_n(row.data<int64_t>() + context - n, n, 1);
                    const auto map = hfa::build_host_mask_map(*ov::get_tensor_impl(row), past, 8);
                    const bool has_past = std::any_of(map.begin(), map.end(), [](auto action) {
                        return action != hfa::HostMaskTileAction::Skip;
                    });
                    auto key = hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row),
                                                            context - query,
                                                            query,
                                                            query,
                                                            query,
                                                            true,
                                                            dtype,
                                                            has_past);
                    ov::Tensor tile(dtype, {1, 1, query, query});
                    hfa::fill_host_mask_tile(key, ov::get_tensor_impl(tile));
                    for (size_t r = 0; r < query; ++r) {
                        for (size_t c = 0; c < query; ++c) {
                            const bool allowed = (r < query - n && !has_past) || (c >= query - n && c <= r);
                            const auto i = r * query + c;
                            const float value = dtype == ov::element::f16
                                                    ? static_cast<float>(tile.data<ov::float16>()[i])
                                                    : tile.data<float>()[i];
                            EXPECT_EQ(value, allowed ? 0 : lowest);
                        }
                    }
                }
            }
        }
    }
}

TEST(HFAHostMaskTest, FinalTilePoolKeyIncludesPastVisibility) {
    auto row = make_host_row(24);
    row.data<int64_t>()[23] = 1;
    const auto no_past =
        hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row), 16, 8, 8, 8, true, ov::element::f32, false);
    const auto with_past =
        hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row), 16, 8, 8, 8, true, ov::element::f32, true);
    EXPECT_FALSE(no_past == with_past);
    hfa::HFARuntimeContext ctx;
    auto allocate = [](const ov::element::Type& dtype, const ov::Shape& shape, const std::string&) {
        return ov::get_tensor_impl(ov::Tensor(dtype, shape));
    };
    auto first = ctx.get_host_mask_tile(no_past, "CPU", allocate);
    auto second = ctx.get_host_mask_tile(with_past, "CPU", allocate);
    EXPECT_NE(first._ptr, second._ptr);
    EXPECT_EQ(first->data<float>()[0], 0);
    EXPECT_EQ(second->data<float>()[0], std::numeric_limits<float>::lowest());
    EXPECT_EQ(ctx.get_host_mask_tile(no_past, "CPU", allocate)._ptr, first._ptr);
}

TEST(HFAHostMaskTest, LegacyPhiAddEqualWindowCoveringContextKeepsGraphMask) {
    for (int64_t window : {24, 25, 262144}) {
        for (bool embeds : {false, true}) {
            std::shared_ptr<ov::Model> model;
            auto sdpa = make_legacy_phi_host_pass_model(model, window, false, embeds);
            ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
            const auto original = sdpa->input_value(3);
            EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
            EXPECT_EQ(sdpa->input_value(3), original);
            EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
        }
    }
}

TEST(HFAHostMaskTest, LegacyPhiNarrowWindowKeepsGraphMask) {
    for (int64_t window : {1, 23}) {
        std::shared_ptr<ov::Model> model;
        auto sdpa = make_legacy_phi_host_pass_model(model, window);
        ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        const auto original = sdpa->input_value(3);
        EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
        EXPECT_EQ(sdpa->input_value(3), original);
        EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
    }
}

TEST(HFAHostMaskTest, LegacyPhiWindowAndImageOrKeepsGraphMask) {
    for (int64_t window : {24, 262144}) {
        std::shared_ptr<ov::Model> model;
        auto sdpa = make_legacy_phi_host_pass_model(model, window, true);
        ASSERT_EQ(sdpa->get_rt_info().at(ov::npuw::NPUW_SDPA_MASK_RT_KEY).as<int64_t>(), window);
        const auto ops = model->get_ops();
        EXPECT_EQ(std::count_if(ops.begin(),
                                ops.end(),
                                [](const auto& node) {
                                    return ov::is_type<ov::op::v13::BitwiseOr>(node);
                                }),
                  2);
        const auto original = sdpa->input_value(3);
        EXPECT_FALSE(ov::npuw::replace_hfa_host_mask(model, true));
        EXPECT_EQ(sdpa->input_value(3), original);
        EXPECT_FALSE(ov::npuw::has_hfa_host_mask(model));
    }
}

TEST(HFAHostMaskTest, CpuTileChainMatchesDoubleReference) {
    using namespace ov;
    constexpr size_t query = 8, past = 24, context = 32, heads = 2, dim = 4;
    constexpr uint32_t seed = 20261023;
    auto row = make_host_row(context, 13);
    std::fill_n(row.data<int64_t>() + context - 3, 3, 1);
    const auto map = hfa::build_host_mask_map(*ov::get_tensor_impl(row), past, query);
    ASSERT_EQ(map,
              (std::vector<hfa::HostMaskTileAction>{hfa::HostMaskTileAction::NoMask,
                                                    hfa::HostMaskTileAction::Mask,
                                                    hfa::HostMaskTileAction::Skip}));
    Core core;
    for (bool fused : {false, true}) {
        for (auto dtype : {element::f32, element::f16}) {
            auto source = build_sdpa_model_transposed_v(query, past, heads, dim);
            if (dtype == element::f16) {
                for (size_t idx = 0; idx < 5; ++idx) {
                    auto param = source->get_parameters()[idx];
                    const auto targets = param->output(0).get_target_inputs();
                    param->set_element_type(dtype);
                    auto cvt = std::make_shared<op::v0::Convert>(param, element::f32);
                    for (const auto& input : targets) {
                        input.replace_source_output(cvt);
                    }
                }
                source->validate_nodes_and_infer_types();
            }
            annotate_mask_rt_info(source, npuw::NPUW_SDPA_MASK_CAUSAL);
            auto off = npuw::function::HostFlashAttention::from(source, fused, false);
            ASSERT_TRUE(off);
            for (const auto& node : source->get_ops()) {
                if (node->get_friendly_name() == "add.0") {
                    node->get_rt_info()[npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
                }
            }
            auto on = npuw::function::HostFlashAttention::from(source, fused, false);
            ASSERT_TRUE(on);
            std::vector<CompiledModel> compiled;
            for (const auto& model : {off->_tile_model,
                                      off->_final_tile_model,
                                      on->_tile_model,
                                      on->_masked_tile_model,
                                      on->_final_tile_model}) {
                compiled.push_back(core.compile_model(model, "CPU", {ov::hint::inference_precision(element::f32)}));
            }
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> distribution(-1, 1);
            auto values = [&](size_t size, float scale) {
                std::vector<double> data(size);
                for (auto& value : data) {
                    const float generated = distribution(rng) * scale;
                    value = dtype == element::f16 ? static_cast<float>(ov::float16(generated)) : generated;
                }
                return data;
            };
            const auto q = values(heads * query * dim, 1 / std::sqrt(static_cast<float>(dim)));
            const auto k = values(heads * context * dim, 1);
            const auto v = values(heads * context * dim, 1);
            auto store = [&](Tensor& tensor, size_t idx, double value) {
                if (tensor.get_element_type() == element::f16) {
                    tensor.data<ov::float16>()[idx] = ov::float16(static_cast<float>(value));
                } else {
                    tensor.data<float>()[idx] = static_cast<float>(value);
                }
            };
            Tensor q_tensor(dtype, {1, heads, query, dim});
            for (size_t i = 0; i < q.size(); ++i) {
                store(q_tensor, i, q[i]);
            }
            auto visible = [&](size_t r, size_t c) {
                return row.data<int64_t>()[c] != 0 && (c < past || c - past <= r);
            };
            std::vector<double> reference(query * heads * dim);
            for (size_t h = 0; h < heads; ++h) {
                for (size_t r = 0; r < query; ++r) {
                    std::vector<double> scores(context);
                    double max = std::numeric_limits<double>::lowest();
                    for (size_t c = 0; c < context; ++c) {
                        if (!visible(r, c)) {
                            continue;
                        }
                        for (size_t d = 0; d < dim; ++d) {
                            scores[c] += q[(h * query + r) * dim + d] * k[(h * context + c) * dim + d];
                        }
                        max = std::max(max, scores[c]);
                    }
                    double sum = 0;
                    for (size_t c = 0; c < context; ++c) {
                        if (!visible(r, c)) {
                            continue;
                        }
                        const auto weight = std::exp(scores[c] - max);
                        sum += weight;
                        for (size_t d = 0; d < dim; ++d) {
                            reference[(r * heads + h) * dim + d] += weight * v[(h * context + c) * dim + d];
                        }
                    }
                    for (size_t d = 0; d < dim; ++d) {
                        reference[(r * heads + h) * dim + d] /= sum;
                    }
                }
            }
            auto run_chain = [&](bool host) {
                std::array<CompiledModel, 3> models{compiled[host ? 2 : 0],
                                                    compiled[host ? 3 : 0],
                                                    compiled[host ? 4 : 1]};
                std::array<InferRequest, 3> requests{models[0].create_infer_request(),
                                                     models[1].create_infer_request(),
                                                     models[2].create_infer_request()};
                std::array<Tensor, 3> state;
                for (size_t i = 0; i < state.size(); ++i) {
                    state[i] = Tensor(dtype, models[0].input(i).get_shape());
                    for (size_t j = 0; j < requests.size(); ++j) {
                        requests[j].set_input_tensor(i, state[i]);
                        if (j != 2) {
                            requests[j].set_output_tensor(i, state[i]);
                        }
                    }
                }
                for (auto& request : requests) {
                    request.set_input_tensor(5, q_tensor);
                }
                hfa::HFARuntimeContext pool;
                auto allocate = [](const element::Type& type, const Shape& shape, const std::string&) {
                    return ov::get_tensor_impl(Tensor(type, shape));
                };
                std::vector<double> result;
                for (size_t repeat = 0; repeat < 2; ++repeat) {
                    for (size_t i = 0; i < state.size(); ++i) {
                        for (size_t e = 0; e < state[i].get_size(); ++e) {
                            store(state[i],
                                  e,
                                  i == 1 ? (dtype == element::f16
                                                ? static_cast<float>(std::numeric_limits<ov::float16>::lowest())
                                                : std::numeric_limits<float>::lowest())
                                         : 0);
                        }
                    }
                    for (size_t offset = 0; offset < context; offset += query) {
                        const bool final = offset == past;
                        const auto action = final ? hfa::HostMaskTileAction::Mask : map[offset / query];
                        if (host && action == hfa::HostMaskTileAction::Skip) {
                            continue;
                        }
                        const size_t id = final ? 2 : host && action == hfa::HostMaskTileAction::NoMask ? 0 : 1;
                        Tensor key(dtype, models[id].input(3).get_shape());
                        Tensor value(dtype, models[id].input(4).get_shape());
                        for (size_t h = 0; h < heads; ++h) {
                            for (size_t c = 0; c < query; ++c) {
                                for (size_t d = 0; d < dim; ++d) {
                                    store(key, (h * query + c) * dim + d, k[(h * context + offset + c) * dim + d]);
                                    store(value, (h * dim + d) * query + c, v[(h * context + offset + c) * dim + d]);
                                }
                            }
                        }
                        requests[id].set_input_tensor(3, key);
                        requests[id].set_input_tensor(4, value);
                        if (models[id].inputs().size() == 7) {
                            Tensor mask(element::f32, {1, 1, query, query});
                            if (host) {
                                auto content = hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row),
                                                                            offset,
                                                                            query,
                                                                            query,
                                                                            query,
                                                                            final,
                                                                            element::f32,
                                                                            true);
                                auto tile = pool.get_host_mask_tile(content, "CPU", allocate);
                                mask = Tensor(element::f32, tile->get_shape(), tile->data());
                            } else {
                                for (size_t r = 0; r < query; ++r) {
                                    for (size_t c = 0; c < query; ++c) {
                                        mask.data<float>()[r * query + c] =
                                            visible(r, offset + c) ? 0 : std::numeric_limits<float>::lowest();
                                    }
                                }
                            }
                            requests[id].set_input_tensor(6, mask);
                        }
                        requests[id].infer();
                    }
                    const auto output = requests[2].get_output_tensor();
                    std::vector<double> current(output.get_size());
                    for (size_t i = 0; i < current.size(); ++i) {
                        current[i] = output.data<const float>()[i];
                        EXPECT_TRUE(std::isfinite(current[i]));
                    }
                    if (repeat != 0) {
                        EXPECT_EQ(current, result);
                    }
                    result = std::move(current);
                }
                return result;
            };
            const auto baseline = run_chain(false);
            const auto hosted = run_chain(true);
            double off_ref = 0, on_ref = 0, off_on = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                off_ref = std::max(off_ref, std::abs(baseline[i] - reference[i]));
                on_ref = std::max(on_ref, std::abs(hosted[i] - reference[i]));
                off_on = std::max(off_on, std::abs(baseline[i] - hosted[i]));
            }
            // CPU f32 past_max I/O aliasing also biases OFF; the round-6 separate-max control removes the error.
            const bool report_only_reference = !fused && dtype == element::f32;
            const auto tolerance = report_only_reference ? 5e-6 : std::max(5e-6, 2 * off_ref);
            std::cout << "HFA_CHAIN fused=" << fused << " state=" << dtype << " seed=" << seed << " off_ref=" << off_ref
                      << " on_ref=" << on_ref << " off_on=" << off_on << " tolerance=" << tolerance << std::endl;
            if (!report_only_reference) {
                EXPECT_LT(off_ref, dtype == element::f16 ? 5e-3 : 1e-4);
                EXPECT_LE(off_ref, tolerance);
                EXPECT_LE(on_ref, tolerance);
            }
            EXPECT_LE(off_on, tolerance);
        }
    }
}

TEST(HFAHostMaskTest, CpuFinalTileIsFiniteWithF16StateAndF32Mask) {
    using namespace ov;
    auto model = build_sdpa_model_mixed_dtype(8, 16, 2, 4);
    for (size_t idx : {3u, 4u}) {
        auto param = model->get_parameters()[idx];
        const auto targets = param->output(0).get_target_inputs();
        param->set_element_type(element::f16);
        auto cvt = std::make_shared<op::v0::Convert>(param, element::f32);
        for (const auto& input : targets) {
            input.replace_source_output(cvt);
        }
    }

    model->validate_nodes_and_infer_types();
    annotate_mask_rt_info(model, npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (node->get_friendly_name() == "add.0") {
            node->get_rt_info()[npuw::NPUW_HFA_HOST_MASK_RT_KEY] = true;
        }
    }
    const auto function = npuw::function::HostFlashAttention::from(model, false);
    ASSERT_TRUE(function);
    EXPECT_EQ(get_input_dtype(function->_final_tile_model, "K_TILE"), element::f16);
    EXPECT_EQ(get_input_dtype(function->_final_tile_model, "V_TILE"), element::f16);
    EXPECT_EQ(get_input_dtype(function->_final_tile_model, "PAST_ACC"), element::f16);
    EXPECT_EQ(get_input_dtype(function->_final_tile_model, "MASK_TILE"), element::f32);
    Core core;
    auto compiled =
        core.compile_model(function->_final_tile_model, "CPU", {ov::hint::inference_precision(element::f32)});
    for (size_t n : {1u, 3u}) {
        auto request = compiled.create_infer_request();
        for (const auto& input : compiled.inputs()) {
            Tensor tensor(input.get_element_type(), input.get_shape());
            std::memset(tensor.data(), 0, tensor.get_byte_size());
            if (input.get_any_name() == "PAST_MAX") {
                std::fill_n(tensor.data<ov::float16>(), tensor.get_size(), std::numeric_limits<ov::float16>::lowest());
            } else if (input.get_any_name() == "V_TILE") {
                std::fill_n(tensor.data<ov::float16>(), tensor.get_size(), ov::float16(0.5f));
            }
            request.set_tensor(input, tensor);
        }
        auto row = make_host_row(24);
        std::fill_n(row.data<int64_t>() + 24 - n, n, 1);
        auto key = hfa::make_host_mask_tile_key(*ov::get_tensor_impl(row), 16, 8, 8, 8, true, element::f32, false);
        Tensor tile(element::f32, {1, 1, 8, 8});
        hfa::fill_host_mask_tile(key, ov::get_tensor_impl(tile));
        request.set_tensor("MASK_TILE", tile);
        request.infer();
        const auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_element_type(), element::f32);
        for (size_t i = 0; i < output.get_size(); ++i) {
            EXPECT_TRUE(std::isfinite(output.data<const float>()[i]));
            EXPECT_NEAR(output.data<const float>()[i], 0.5f, 1e-3);
        }
    }
}
