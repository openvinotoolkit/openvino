// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_flash_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>

#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"

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
                                            size_t head_dim = HEAD_DIM,
                                            bool with_attention_sink = false,
                                            bool with_post_qk_scale = false,
                                            bool post_qk_scale_is_constant = false) {
    using namespace ov;

    const size_t context_size = past_len + query_size;
    const Shape past_shape = {BATCH, num_heads, past_len, head_dim};
    const Shape new_shape = {BATCH, num_heads, query_size, head_dim};
    const Shape mask_shape = {BATCH, 1, query_size, context_size};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    auto query = make_param("query.0", new_shape);
    auto past_key = make_param("past_key_values.0.key", past_shape);
    auto past_val = make_param("past_key_values.0.value", past_shape);
    auto new_key = make_param("new_key.0", new_shape);
    auto new_val = make_param("new_value.0", new_shape);
    auto mask = make_param("mask.0", mask_shape);
    std::shared_ptr<op::v0::Parameter> attention_sink;
    if (with_attention_sink) {
        attention_sink = make_param("attention_sink.0", Shape{BATCH, num_heads, 1, 1});
    }
    std::shared_ptr<ov::Node> attention_scale;
    if (with_post_qk_scale) {
        if (post_qk_scale_is_constant) {
            attention_scale = op::v0::Constant::create(element::f32, Shape{}, {0.5f});
        } else {
            attention_scale = make_param("attention_scale.0", Shape{});
        }
    }

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
    key_concat->set_friendly_name("concat_key.0");
    auto val_concat = std::make_shared<op::v0::Concat>(OutputVector{past_val, new_val}, 2);
    val_concat->set_friendly_name("concat_value.0");

    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    Output<Node> scores = qk;
    if (attention_scale) {
        scores = std::make_shared<op::v1::Multiply>(scores, attention_scale->output(0));
    }
    auto add = std::make_shared<op::v1::Add>(scores, mask->output(0));
    add->set_friendly_name("add.0");
    Output<Node> softmax_input = add;
    if (attention_sink) {
        auto sink_shape = op::v0::Constant::create(element::i64,
                                                   Shape{4},
                                                   std::vector<int64_t>{static_cast<int64_t>(BATCH),
                                                                        static_cast<int64_t>(num_heads),
                                                                        static_cast<int64_t>(query_size),
                                                                        1});
        auto sink_broadcast = std::make_shared<op::v1::Broadcast>(attention_sink, sink_shape);
        softmax_input = std::make_shared<op::v0::Concat>(OutputVector{add, sink_broadcast}, -1);
    }
    auto softmax = std::make_shared<op::v8::Softmax>(softmax_input, 3);
    softmax->set_friendly_name("softmax.0");
    Output<Node> probabilities = softmax;
    if (attention_sink) {
        probabilities = std::make_shared<op::v8::Slice>(
            softmax,
            op::v0::Constant::create(element::i64, Shape{1}, {0}),
            op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(context_size)}),
            op::v0::Constant::create(element::i64, Shape{1}, {1}),
            op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    }
    auto matmul2 = std::make_shared<op::v0::MatMul>(probabilities, val_concat->output(0));
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

    ParameterVector params;
    ResultVector results;
    auto make_param = [&](const std::string& name, const Shape& shape, element::Type dtype) {
        auto p = std::make_shared<op::v0::Parameter>(dtype, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    // Q is f32 (compute precision), KV cache stored as f16 (storage precision),
    // present-KV from the upstream NPU subgraph is f32.
    // This mirrors the real Gemma-4 pattern:
    //   Convert(f16 past_block) ─┐
    //   f32 present_kv           ┴→ Concat(f32) → MatMul
    auto query = make_param("query.0", q_shape_s, element::f32);
    auto past_key = make_param("past_key_values.0.key", kv_shape, element::f16);
    auto past_val = make_param("past_key_values.0.value", kv_shape, element::f16);
    auto new_key = make_param("new_key.0", new_kv_shape, element::f32);
    auto new_val = make_param("new_value.0", new_kv_shape, element::f32);
    auto mask = make_param("mask.0", mask_shape, element::f32);

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

TEST(HostFlashAttentionFromTest, SupportsAttentionSinkWithPostQKScale) {
    auto result = ov::npuw::function::HostFlashAttention::from(
        build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true, true),
        false);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->_attention_scale_param_idx.has_value());
    EXPECT_TRUE(result->_attention_sink_param_idx.has_value());
    EXPECT_NE(result->_tile_param_index_map.find(ov::npuw::HFATileInputId::SCALE), result->_tile_param_index_map.end());
    EXPECT_TRUE(
        std::any_of(result->_tile_model->inputs().begin(), result->_tile_model->inputs().end(), [](const auto& input) {
            return input.get_names().count("SCALE") != 0;
        }));
    EXPECT_TRUE(std::any_of(result->_final_tile_model->inputs().begin(),
                            result->_final_tile_model->inputs().end(),
                            [](const auto& input) {
                                return input.get_names().count("SCALE") != 0;
                            }));
}

TEST(HostFlashAttentionFromTest, EmbedsConstantPostQKScale) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, false, true, true);

    auto result = ov::npuw::function::HostFlashAttention::from(model, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->_attention_scale_param_idx.has_value());
    const auto tile_inputs = result->_tile_model->inputs();
    const auto final_tile_inputs = result->_final_tile_model->inputs();
    EXPECT_TRUE(std::none_of(tile_inputs.begin(), tile_inputs.end(), [](const auto& input) {
        return input.get_names().count("SCALE") != 0;
    }));
    EXPECT_TRUE(std::none_of(final_tile_inputs.begin(), final_tile_inputs.end(), [](const auto& input) {
        return input.get_names().count("SCALE") != 0;
    }));
    const auto tile_ops = result->_tile_model->get_ops();
    EXPECT_TRUE(std::any_of(tile_ops.begin(), tile_ops.end(), [](const auto& node) {
        return node->get_friendly_name() == "q_scaled" &&
               ov::is_type<ov::op::v0::Constant>(node->input_value(1).get_node_shared_ptr());
    }));
}

TEST(HostFlashAttentionFromTest, FusedScaleAndSinkMaskSkippingKeepsScaleWithoutMask) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true, true);
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);

    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);

    ASSERT_TRUE(result.has_value());
    const auto has_input = [](const std::shared_ptr<ov::Model>& tile_model, const std::string& name) {
        const auto& inputs = tile_model->inputs();
        return std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) {
            return input.get_names().count(name) != 0;
        });
    };
    EXPECT_TRUE(has_input(result->_tile_model, "SCALE"));
    EXPECT_FALSE(has_input(result->_tile_model, "MASK_TILE"));
    EXPECT_TRUE(has_input(result->_final_tile_model, "SCALE"));
    EXPECT_TRUE(has_input(result->_final_tile_model, "MASK_TILE"));
    EXPECT_EQ(result->_tile_param_index_map.at(ov::npuw::HFATileInputId::SCALE), 6u);
    EXPECT_EQ(result->_final_tile_param_index_map.at(ov::npuw::HFATileInputId::SCALE), 7u);

    const ov::npuw::compiled::HostFlashAttention compiled_hfa(*result);
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._tile_input_indices.scale, std::optional<std::size_t>{6u});
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._final_tile_input_indices.scale, std::optional<std::size_t>{7u});
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._final_tile_input_indices.mask, 6u);
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
    EXPECT_EQ(result->_tile_param_index_map.size(), static_cast<size_t>(ov::npuw::HFATileInputId::COUNT) - 1u);
    EXPECT_EQ(result->_tile_param_index_map.count(ov::npuw::HFATileInputId::SCALE), 0u);
}

TEST(HostFlashAttentionFromTest, Fused_ParamIndexMapHasAllSevenEntries) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_param_index_map.size(), static_cast<size_t>(ov::npuw::HFATileInputId::COUNT) - 1u);
    EXPECT_EQ(result->_tile_param_index_map.count(ov::npuw::HFATileInputId::SCALE), 0u);
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
    EXPECT_EQ(result->_tile_size, static_cast<int64_t>(QUERY_SIZE));
    EXPECT_EQ(result->_query_size, QUERY_SIZE);
}

TEST(HostFlashAttentionFromTest, Fused_ContextSizeIsCorrect) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_context_size, QUERY_SIZE + PAST_LEN);
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

    ParameterVector params;
    ResultVector results;
    auto make_param = [&](const std::string& name, const Shape& shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    auto query = make_param("query.0", q_shape);
    auto past_key = make_param("past_key_values.0.key", past_k_shape);
    auto past_val = make_param("past_key_values.0.value", past_v_shape);
    auto new_key = make_param("new_key.0", new_k_shape);
    auto new_val = make_param("new_value.0", new_v_shape);
    auto mask = make_param("mask.0", mask_shape);

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
