// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_flash_attention.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "intel_npu/ops/flash_attention_tile.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"
#include "variable_state.hpp"

#define private public
#include "compiled_model.hpp"
#undef private
#include "attn/attn_subgraph.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

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

    ResultVector results;

    ParameterVector params;
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

TEST(HostFlashAttentionFromTest, SupportsAttentionSinkWithPostQKScale) {
    for (const bool fused_flash_attention : {false, true}) {
        SCOPED_TRACE(fused_flash_attention);
        auto result = ov::npuw::function::HostFlashAttention::from(
            build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true, true),
            fused_flash_attention);

        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result->_attention_scale_param_idx.has_value());
        EXPECT_TRUE(result->_attention_sink_param_idx.has_value());
        EXPECT_NE(result->_tile_param_index_map.find(ov::npuw::HFATileInputId::SCALE),
                  result->_tile_param_index_map.end());
        const auto tile_inputs = result->_tile_model->inputs();
        const auto final_tile_inputs = result->_final_tile_model->inputs();
        EXPECT_TRUE(std::any_of(tile_inputs.begin(), tile_inputs.end(), [](const auto& input) {
            return input.get_names().count("SCALE") != 0;
        }));
        EXPECT_TRUE(std::any_of(final_tile_inputs.begin(), final_tile_inputs.end(), [](const auto& input) {
            return input.get_names().count("SCALE") != 0;
        }));
    }
}

TEST(HostFlashAttentionFromTest, EmbedsConstantPostQKScale) {
    for (const bool fused_flash_attention : {false, true}) {
        SCOPED_TRACE(fused_flash_attention);
        auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, false, true, true);
        auto result = ov::npuw::function::HostFlashAttention::from(model, fused_flash_attention);

        ASSERT_TRUE(result.has_value());
        EXPECT_FALSE(result->_attention_scale_param_idx.has_value());
        for (const auto& tile_model : {result->_tile_model, result->_final_tile_model}) {
            const auto tile_inputs = tile_model->inputs();
            EXPECT_TRUE(std::none_of(tile_inputs.begin(), tile_inputs.end(), [](const auto& input) {
                return input.get_names().count("SCALE") != 0;
            }));
            const auto tile_ops = tile_model->get_ops();
            const auto scaled_query = std::find_if(tile_ops.begin(), tile_ops.end(), [](const auto& node) {
                return node->get_friendly_name() == "q_scaled";
            });
            ASSERT_NE(scaled_query, tile_ops.end());
            const auto scale_constant =
                ov::as_type_ptr<ov::op::v0::Constant>((*scaled_query)->input_value(1).get_node_shared_ptr());
            ASSERT_TRUE(scale_constant);
            EXPECT_EQ(scale_constant->cast_vector<float>(), std::vector<float>{0.5f});
        }
    }
}

TEST(HostFlashAttentionFromTest, PreservesPostQKScaleConversions) {
    for (const bool constant_scale : {false, true}) {
        for (const bool fused_flash_attention : {false, true}) {
            SCOPED_TRACE(constant_scale);
            SCOPED_TRACE(fused_flash_attention);
            auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, false, true, constant_scale);
            const auto pattern = ov::npuw::util::find_sdpa_pattern_nodes(model);
            ASSERT_TRUE(pattern.attention_scale_node);
            const auto scale_consumers = pattern.attention_scale_node->output(0).get_target_inputs();
            std::shared_ptr<ov::Node> scale_source = pattern.attention_scale_node;
            if (constant_scale) {
                scale_source = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-8f});
            }
            const auto half = std::make_shared<ov::op::v0::Convert>(scale_source, ov::element::f16);
            const auto converted = std::make_shared<ov::op::v0::Convert>(half, ov::element::f32);
            for (auto input : scale_consumers) {
                input.replace_source_output(converted);
            }
            model->validate_nodes_and_infer_types();

            const auto result = ov::npuw::function::HostFlashAttention::from(model, fused_flash_attention);
            ASSERT_TRUE(result.has_value());
            for (const auto& tile_model : {result->_tile_model, result->_final_tile_model}) {
                const auto tile_ops = tile_model->get_ops();
                const auto scaled_query = std::find_if(tile_ops.begin(), tile_ops.end(), [](const auto& node) {
                    return node->get_friendly_name() == "q_scaled";
                });
                ASSERT_NE(scaled_query, tile_ops.end());
                ov::ParameterVector parameters;
                ov::TensorVector inputs;
                if (!constant_scale) {
                    const auto tile_parameters = tile_model->get_parameters();
                    const auto scale_parameter =
                        std::find_if(tile_parameters.begin(), tile_parameters.end(), [](const auto& parameter) {
                            return parameter->get_friendly_name() == "SCALE";
                        });
                    ASSERT_NE(scale_parameter, tile_parameters.end());
                    parameters.push_back(*scale_parameter);
                    inputs.emplace_back(ov::element::f32, ov::Shape{});
                    inputs.back().data<float>()[0] = 1e-8f;
                }
                ov::Model scale_model(ov::OutputVector{(*scaled_query)->input_value(1)}, parameters);
                ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{})};
                ASSERT_TRUE(scale_model.evaluate(outputs, inputs));
                EXPECT_FLOAT_EQ(outputs.front().data<const float>()[0], 0.0f);
            }
        }
    }
}

TEST(HostFlashAttentionFromTest, SupportsFunctionClosureInputs) {
    for (const bool fused_flash_attention : {false, true}) {
        auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true, true);
        const auto num_inputs = model->get_parameters().size();
        const auto hfa = ov::npuw::function::HostFlashAttention::from(model, fused_flash_attention);
        ASSERT_TRUE(hfa);
        ASSERT_TRUE(hfa->_attention_sink_param_idx);
        ASSERT_TRUE(hfa->_attention_scale_param_idx);
        EXPECT_GE(*hfa->_attention_sink_param_idx, num_inputs - 2);
        EXPECT_GE(*hfa->_attention_scale_param_idx, num_inputs - 2);
    }
}

TEST(HostFlashAttentionFromTest, RejectsGenerateAttention) {
    for (const bool with_sink : {false, true}) {
        auto model = build_sdpa_model(1, PAST_LEN, NUM_HEADS, HEAD_DIM, with_sink);
        ASSERT_TRUE(ov::npuw::util::find_sdpa_pattern_nodes(model).is_valid());
        EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, false).has_value());
        EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, true).has_value());
    }
}

TEST(HostFlashAttentionFromTest, RejectsConstantAttentionSink) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true);
    auto sink = model->get_parameters().back();
    auto constant = ov::op::v0::Constant::create(ov::element::f32, sink->get_shape(), {1.0f});
    for (auto input : sink->output(0).get_target_inputs()) {
        input.replace_source_output(constant);
    }
    model->remove_parameter(sink);
    model->validate_nodes_and_infer_types();

    ASSERT_TRUE(ov::npuw::util::find_sdpa_pattern_nodes(model).is_valid());
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, false).has_value());
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, true).has_value());
}

TEST(HostFlashAttentionFromTest, RejectsReshapedAttentionSink) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true);
    auto sink = model->get_parameters().back();
    const auto consumers = sink->output(0).get_target_inputs();
    auto shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{static_cast<int64_t>(BATCH), static_cast<int64_t>(NUM_HEADS), 1, 1});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(sink, shape, false);
    for (auto input : consumers) {
        input.replace_source_output(reshaped);
    }
    model->validate_nodes_and_infer_types();

    ASSERT_TRUE(ov::npuw::util::find_sdpa_pattern_nodes(model).is_valid());
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, false).has_value());
    EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, true).has_value());
}

TEST(HostFlashAttentionFromTest, AcceptsConvertedParameterAttentionSink) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true);
    auto sink = model->get_parameters().back();
    const auto consumers = sink->output(0).get_target_inputs();
    auto half = std::make_shared<ov::op::v0::Convert>(sink, ov::element::f16);
    auto converted = std::make_shared<ov::op::v0::Convert>(half, ov::element::f32);
    for (auto input : consumers) {
        input.replace_source_output(converted);
    }
    model->validate_nodes_and_infer_types();

    for (const bool fused : {false, true}) {
        auto result = ov::npuw::function::HostFlashAttention::from(model, fused);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->_attention_sink_param_idx,
                  std::optional<std::size_t>{static_cast<std::size_t>(model->get_parameter_index(sink))});
        EXPECT_TRUE(result->_attention_sink_round_to_f16);
        EXPECT_TRUE(ov::npuw::compiled::HostFlashAttention(*result)._sdpa_attention_info._attention_sink_round_to_f16);
    }
}

TEST(HostFlashAttentionFromTest, ConvertedSinkMatchesReferenceAcrossTiles) {
    for (const bool fused : {false, true}) {
        SCOPED_TRACE(fused);
        auto model = build_sdpa_model(2, 2, 1, 1, true);
        auto sink = model->get_parameters().back();
        const auto consumers = sink->output(0).get_target_inputs();
        auto half = std::make_shared<ov::op::v0::Convert>(sink, ov::element::f16);
        auto converted = std::make_shared<ov::op::v0::Convert>(half, ov::element::f32);
        for (auto input : consumers) {
            input.replace_source_output(converted);
        }
        model->validate_nodes_and_infer_types();

        const auto hfa = ov::npuw::function::HostFlashAttention::from(model, fused);
        ASSERT_TRUE(hfa);
        const ov::npuw::compiled::HostFlashAttention compiled_hfa(*hfa);
        ASSERT_TRUE(compiled_hfa._sdpa_attention_info._attention_sink_round_to_f16);
        using InputId = ov::npuw::HFATileInputId;
        using OutputId = ov::npuw::HFATileOutputId;
        const auto make_tile_inputs = [](const std::shared_ptr<ov::Model>& tile,
                                         const std::map<InputId, size_t>& indices) {
            ov::TensorVector tensors;
            for (const auto& input : tile->inputs()) {
                tensors.emplace_back(input.get_element_type(), input.get_shape());
                std::fill_n(tensors.back().data<float>(), tensors.back().get_size(), 0.0f);
            }
            for (const auto& entry : {std::make_pair(InputId::Q, 1.0f),
                                      std::make_pair(InputId::K_TILE, 2048.0f),
                                      std::make_pair(InputId::V_TILE, 1.0f)}) {
                auto& tensor = tensors.at(indices.at(entry.first));
                std::fill_n(tensor.data<float>(), tensor.get_size(), entry.second);
            }
            return tensors;
        };
        auto regular_inputs = make_tile_inputs(hfa->_tile_model, hfa->_tile_param_index_map);
        auto acc = ov::get_tensor_impl(regular_inputs.at(hfa->_tile_param_index_map.at(InputId::PAST_ACC)));
        auto max = ov::get_tensor_impl(regular_inputs.at(hfa->_tile_param_index_map.at(InputId::PAST_MAX)));
        auto sum = ov::get_tensor_impl(regular_inputs.at(hfa->_tile_param_index_map.at(InputId::PAST_D)));
        ov::Tensor sink_tensor(ov::element::f32, sink->get_shape());
        sink_tensor.data<float>()[0] = 2049.0f;
        ov::npuw::runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(
            acc,
            max,
            sum,
            ov::get_tensor_impl(sink_tensor),
            compiled_hfa._sdpa_attention_info._attention_sink_round_to_f16);
        ov::TensorVector regular_outputs;
        for (const auto& output : hfa->_tile_model->outputs()) {
            regular_outputs.emplace_back(output.get_element_type(), output.get_shape());
        }
        ASSERT_TRUE(hfa->_tile_model->evaluate(regular_outputs, regular_inputs));

        auto final_inputs = make_tile_inputs(hfa->_final_tile_model, hfa->_final_tile_param_index_map);
        final_inputs.at(hfa->_final_tile_param_index_map.at(InputId::PAST_ACC)) =
            regular_outputs.at(hfa->_tile_output_index_map.at(OutputId::ACC));
        final_inputs.at(hfa->_final_tile_param_index_map.at(InputId::PAST_MAX)) =
            regular_outputs.at(hfa->_tile_output_index_map.at(OutputId::MAXX));
        final_inputs.at(hfa->_final_tile_param_index_map.at(InputId::PAST_D)) =
            regular_outputs.at(hfa->_tile_output_index_map.at(OutputId::D));
        ov::TensorVector actual{ov::Tensor(ov::element::f32, hfa->_final_tile_model->output().get_shape())};
        ASSERT_TRUE(hfa->_final_tile_model->evaluate(actual, final_inputs));

        ov::TensorVector inputs;
        for (const auto& input : model->inputs()) {
            inputs.emplace_back(input.get_element_type(), input.get_shape());
            const auto& name = input.get_any_name();
            float value = 1.0f;
            if (name.find("attention_sink") == 0) {
                value = 2049.0f;
            } else if (name == "past_key_values.0.key" || name == "new_key.0") {
                value = 2048.0f;
            } else if (name.find("mask") == 0) {
                value = 0.0f;
            }
            std::fill_n(inputs.back().data<float>(), inputs.back().get_size(), value);
        }
        ov::TensorVector expected;
        for (const auto& output : model->outputs()) {
            expected.emplace_back(output.get_element_type(), output.get_shape());
        }
        ASSERT_TRUE(model->evaluate(expected, inputs));
        for (size_t index = 0; index < actual.front().get_size(); ++index) {
            EXPECT_NEAR(expected.back().data<const float>()[index], 0.8f, 1e-6f);
            EXPECT_NEAR(actual.front().data<const float>()[index], expected.back().data<const float>()[index], 1e-6f);
        }
    }
}

TEST(HostFlashAttentionFromTest, RejectsNonFloatingAttentionSinkConversions) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true);
    auto sink = model->get_parameters().back();
    const auto consumers = sink->output(0).get_target_inputs();
    auto integer = std::make_shared<ov::op::v0::Convert>(sink, ov::element::i32);
    auto converted = std::make_shared<ov::op::v0::Convert>(integer, ov::element::f32);
    for (auto input : consumers) {
        input.replace_source_output(converted);
    }
    model->validate_nodes_and_infer_types();
    for (const bool fused : {false, true}) {
        EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, fused).has_value());
    }
}

TEST(HostFlashAttentionFromTest, RejectsNonNumpyAttentionSinkBroadcast) {
    for (const bool version3 : {false, true}) {
        SCOPED_TRACE(version3);
        auto model = build_sdpa_model(2, 4, 2, HEAD_DIM, true);
        auto old_sink = model->get_parameters().back();
        auto old_broadcast = old_sink->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        auto sink = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
        auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 2, 1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 3});
        std::shared_ptr<ov::Node> broadcast;
        if (version3) {
            broadcast = std::make_shared<ov::op::v3::Broadcast>(sink, shape, axes);
        } else {
            broadcast =
                std::make_shared<ov::op::v1::Broadcast>(sink,
                                                        shape,
                                                        axes,
                                                        ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NONE));
        }
        for (auto input : old_broadcast->output(0).get_target_inputs()) {
            input.replace_source_output(broadcast);
        }
        model->remove_parameter(old_sink);
        model->add_parameters({sink});
        model->validate_nodes_and_infer_types();

        ov::Tensor input(ov::element::f32, sink->get_shape());
        input.data<float>()[0] = 0.0f;
        input.data<float>()[1] = 2.0f;
        ov::Model broadcast_model(ov::OutputVector{broadcast}, ov::ParameterVector{sink});
        ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, 2, 2, 1})};
        ASSERT_TRUE(broadcast_model.evaluate(outputs, ov::TensorVector{input}));
        EXPECT_EQ(std::vector<float>(outputs.front().data<const float>(), outputs.front().data<const float>() + 4),
                  (std::vector<float>{0.0f, 0.0f, 2.0f, 2.0f}));
        ASSERT_TRUE(ov::npuw::util::find_sdpa_pattern_nodes(model).is_valid());
        EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, false).has_value());
        EXPECT_FALSE(ov::npuw::function::HostFlashAttention::from(model, true).has_value());
    }
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
    ASSERT_TRUE(result->_masked_tile_model);
    EXPECT_TRUE(has_input(result->_masked_tile_model, "SCALE"));
    EXPECT_TRUE(has_input(result->_masked_tile_model, "MASK_TILE"));
    EXPECT_TRUE(has_input(result->_final_tile_model, "SCALE"));
    EXPECT_TRUE(has_input(result->_final_tile_model, "MASK_TILE"));
    EXPECT_EQ(result->_tile_param_index_map.at(ov::npuw::HFATileInputId::SCALE), 6u);
    EXPECT_EQ(result->_final_tile_param_index_map.at(ov::npuw::HFATileInputId::SCALE), 7u);

    const ov::npuw::compiled::HostFlashAttention compiled_hfa(*result);
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._tile_input_indices.scale, std::optional<std::size_t>{6u});
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._final_tile_input_indices.scale, std::optional<std::size_t>{7u});
    EXPECT_EQ(compiled_hfa._sdpa_attention_info._final_tile_input_indices.mask, 6u);
}

TEST(HostFlashAttentionFromTest, MaskedTilePreservesConstantPostQKScale) {
    auto model = build_sdpa_model(QUERY_SIZE, PAST_LEN, NUM_HEADS, HEAD_DIM, true, true, true);
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto result = ov::npuw::function::HostFlashAttention::from(model, true, true);

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->_masked_tile_model);
    const auto tile_ops = result->_masked_tile_model->get_ops();
    const auto scaled_query = std::find_if(tile_ops.begin(), tile_ops.end(), [](const auto& node) {
        return node->get_friendly_name() == "q_scaled";
    });
    ASSERT_NE(scaled_query, tile_ops.end());
    const auto scale_constant =
        ov::as_type_ptr<ov::op::v0::Constant>((*scaled_query)->input_value(1).get_node_shared_ptr());
    ASSERT_TRUE(scale_constant);
    EXPECT_EQ(scale_constant->cast_vector<float>(), std::vector<float>{0.5f});
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

// Minimal ISyncInferRequest: the find()s only read get_inputs() and query_state(), so no tensors are needed.
class PosIdsFakeInferRequest final : public ov::ISyncInferRequest {
public:
    explicit PosIdsFakeInferRequest(const std::shared_ptr<const ov::ICompiledModel>& cm,
                                    std::vector<ov::SoPtr<ov::IVariableState>> states = {})
        : ov::ISyncInferRequest(cm),
          m_states(std::move(states)) {}

    void infer() override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return m_states;
    }
    void check_tensors() const override {}

private:
    std::vector<ov::SoPtr<ov::IVariableState>> m_states;
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

// A request for a model with no position_ids input (like a LiteRT prefill graph), exposing the
// given variable states.
std::shared_ptr<ov::ISyncInferRequest> make_no_position_ids_request(std::vector<ov::SoPtr<ov::IVariableState>> states) {
    auto embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 8});
    embeds->set_friendly_name("embeddings");
    embeds->output(0).get_tensor().set_names({"embeddings"});
    auto result = std::make_shared<ov::op::v0::Result>(embeds);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{embeds});

    auto plugin = std::make_shared<PosIdsNullPlugin>();
    auto compiled = std::make_shared<PosIdsStubCompiledModel>(model, plugin);
    return std::make_shared<PosIdsFakeInferRequest>(compiled, std::move(states));
}

ov::SoPtr<ov::IVariableState> make_history_state(int64_t value) {
    auto state = std::make_shared<ov::npuw::HistorySizeState>();
    state->set(value);
    return {state, nullptr};
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

// ---------------------------------------------------------------------------
// runtime::host_flash_attention::HistorySize
// ---------------------------------------------------------------------------

TEST(HostFlashAttentionHistorySizeTest, PrefillWithoutHistoryHasNoPast) {
    ov::npuw::runtime::host_flash_attention::HistorySize selector(1024);
    selector.prepare(0);
    EXPECT_EQ(selector.this_case(), ov::npuw::runtime::host_flash_attention::Selector::Case::PREFILL);
    EXPECT_EQ(selector.context_length(), 1024);
}

TEST(HostFlashAttentionHistorySizeTest, PrefillRoundsHistoryUpToWholeChunks) {
    ov::npuw::runtime::host_flash_attention::HistorySize selector(1024);
    selector.prepare(1024);
    EXPECT_EQ(selector.context_length(), 1024 + 1024);
    // A partly filled last chunk (e.g. LiteRT-LM's 1023-token first chunk) still needs its whole tile.
    selector.prepare(1023);
    EXPECT_EQ(selector.context_length(), 1024 + 1024);
    selector.prepare(2100);
    EXPECT_EQ(selector.context_length(), 3 * 1024 + 1024);
}

TEST(HostFlashAttentionHistorySizeTest, FollowsEveryPrepare) {
    ov::npuw::runtime::host_flash_attention::HistorySize selector(16);
    selector.prepare(48);
    EXPECT_EQ(selector.context_length(), 48 + 16);
    selector.prepare(0);  // a new conversation
    EXPECT_EQ(selector.context_length(), 16);
}

TEST(HostFlashAttentionHistorySizeFindTest, MatchesWhenHistoryIsSet) {
    auto rq = make_no_position_ids_request({make_history_state(0)});
    EXPECT_EQ(ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq), nullptr);
    auto selector = ov::npuw::runtime::host_flash_attention::HistorySize::find(16, *rq);
    ASSERT_NE(selector, nullptr);
    EXPECT_EQ(selector->this_case(), ov::npuw::runtime::host_flash_attention::Selector::Case::PREFILL);
}

// A host that never sets the history must get an error, not HFA silently running without the past.
TEST(HostFlashAttentionHistorySizeFindTest, RejectsUnsetHistory) {
    auto rq = make_no_position_ids_request({make_history_state(ov::npuw::HistorySizeState::kUnset)});
    EXPECT_EQ(ov::npuw::runtime::host_flash_attention::HistorySize::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionHistorySizeFindTest, RejectsMissingHistoryState) {
    auto rq = make_no_position_ids_request({});
    EXPECT_EQ(ov::npuw::runtime::host_flash_attention::HistorySize::find(16, *rq), nullptr);
}

TEST(HostFlashAttentionHistorySizeTest, GenerateTakesHistoryAsIs) {
    ov::npuw::runtime::host_flash_attention::HistorySize selector(1);
    selector.prepare(37);
    EXPECT_EQ(selector.this_case(), ov::npuw::runtime::host_flash_attention::Selector::Case::GENERATE);
    EXPECT_EQ(selector.context_length(), 37 + 1);
}

TEST(HostFlashAttentionHistorySizeTest, PreservesExactHistory) {
    ov::npuw::runtime::host_flash_attention::HistorySize selector(16);
    for (const int64_t history : {19, 31, 32, 0}) {
        selector.prepare(history);
        EXPECT_EQ(selector.history_length(), history);
        EXPECT_EQ(selector.context_length(), 16 + ((history + 15) / 16) * 16);
    }
}

TEST(HostFlashAttentionPositionIdsFindTest, PreservesExactHistory) {
    auto rq = make_position_ids_request({1, 16});
    ov::Tensor ids(ov::element::i64, {1, 16});
    std::iota(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size(), int64_t{19});
    rq->set_tensor(rq->get_inputs()[0], ov::get_tensor_impl(ids));
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(16, *rq);
    ASSERT_NE(selector, nullptr);
    selector->prepare(19);
    EXPECT_EQ(selector->history_length(), 19);
    EXPECT_EQ(selector->context_length(), 48);
}

TEST(HostFlashAttentionPositionIdsFindTest, PrefillZeroIdsWithoutHistoryHasNoPast) {
    auto rq = make_position_ids_request({1, QUERY_SIZE});
    ov::Tensor ids(ov::element::i64, {1, QUERY_SIZE});
    std::fill_n(ids.data<int64_t>(), ids.get_size(), int64_t{0});
    rq->set_tensor(rq->get_inputs()[0], ov::get_tensor_impl(ids));
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(QUERY_SIZE, *rq);
    ASSERT_NE(selector, nullptr);
    selector->prepare(0);
    EXPECT_EQ(selector->context_length(), QUERY_SIZE);
}

TEST(HostFlashAttentionPositionIdsFindTest, PrefillZeroIdsAfterResetHasNoStalePast) {
    auto rq = make_position_ids_request({1, QUERY_SIZE});
    ov::Tensor ids(ov::element::i64, {1, QUERY_SIZE});
    const int64_t history = 2 * QUERY_SIZE + 3;
    std::iota(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size(), history);
    rq->set_tensor(rq->get_inputs()[0], ov::get_tensor_impl(ids));
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(QUERY_SIZE, *rq);
    ASSERT_NE(selector, nullptr);
    selector->prepare(history);
    EXPECT_EQ(selector->context_length(), 4 * QUERY_SIZE);

    std::fill_n(ids.data<int64_t>(), ids.get_size(), int64_t{0});
    selector->prepare(0);
    EXPECT_EQ(selector->context_length(), QUERY_SIZE);
}

TEST(HostFlashAttentionPositionIdsFindTest, PrefillZeroIdsWithHistoryThrows) {
    auto rq = make_position_ids_request({1, QUERY_SIZE});
    ov::Tensor ids(ov::element::i64, {1, QUERY_SIZE});
    std::fill_n(ids.data<int64_t>(), ids.get_size(), int64_t{0});
    rq->set_tensor(rq->get_inputs()[0], ov::get_tensor_impl(ids));
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(QUERY_SIZE, *rq);
    ASSERT_NE(selector, nullptr);
    EXPECT_THROW(selector->prepare(19), ov::Exception);
}

TEST(HostFlashAttentionPositionIdsFindTest, GenerateZeroIdsWithoutHistoryStartsAtPositionZero) {
    auto rq = make_position_ids_request({1, 1});
    ov::Tensor ids(ov::element::i64, {1, 1});
    std::fill_n(ids.data<int64_t>(), ids.get_size(), int64_t{0});
    rq->set_tensor(rq->get_inputs()[0], ov::get_tensor_impl(ids));
    auto selector = ov::npuw::runtime::host_flash_attention::PositionIDs::find(1, *rq);
    ASSERT_NE(selector, nullptr);
    selector->prepare(0);
    EXPECT_EQ(selector->context_length(), 1);
}

namespace {

class HFACpuCompiledModel;

class HFACpuInferRequest final : public ov::ISyncInferRequest {
public:
    HFACpuInferRequest(const std::shared_ptr<const HFACpuCompiledModel>& model, ov::InferRequest request);
    void infer() override;
    void check_tensors() const override {}
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }

private:
    ov::InferRequest m_cpu;
};

class HFACpuCompiledModel final : public ov::ICompiledModel {
public:
    HFACpuCompiledModel(const std::shared_ptr<const ov::Model>& model,
                        const std::shared_ptr<const ov::IPlugin>& plugin,
                        ov::CompiledModel cpu,
                        size_t id)
        : ov::ICompiledModel(model, plugin),
          m_model(model),
          m_cpu(std::move(cpu)),
          m_id(id) {}
    void export_model(std::ostream& stream) const override {
        ov::npuw::s11n::write(stream, m_id);
        m_cpu.export_model(stream);
    }
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string& name) const override {
        return m_cpu.get_property(name);
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const HFACpuCompiledModel>(shared_from_this());
        return std::make_shared<HFACpuInferRequest>(self, m_cpu.create_infer_request());
    }
    mutable std::atomic<size_t> infer_count{0};

private:
    std::shared_ptr<const ov::Model> m_model;
    mutable ov::CompiledModel m_cpu;
    size_t m_id;
};

HFACpuInferRequest::HFACpuInferRequest(const std::shared_ptr<const HFACpuCompiledModel>& model,
                                       ov::InferRequest request)
    : ov::ISyncInferRequest(model),
      m_cpu(std::move(request)) {
    for (const auto& port : model->inputs()) {
        set_tensor(port, ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape())));
    }
    for (const auto& port : model->outputs()) {
        set_tensor(port, ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape())));
    }
}

void HFACpuInferRequest::infer() {
    const auto model = std::static_pointer_cast<const HFACpuCompiledModel>(get_compiled_model());
    for (size_t i = 0; i < model->inputs().size(); ++i) {
        m_cpu.set_input_tensor(i, ov::make_tensor(get_tensor(model->inputs()[i])));
    }
    for (size_t i = 0; i < model->outputs().size(); ++i) {
        m_cpu.set_output_tensor(i, ov::make_tensor(get_tensor(model->outputs()[i])));
    }
    m_cpu.infer();
    ++model->infer_count;
}

std::shared_ptr<ov::Model> build_hfa_runtime_model(size_t past_capacity = PAST_LEN,
                                                   bool transposed_v = false,
                                                   bool with_sink_and_scale = false) {
    auto sdpa = transposed_v ? build_sdpa_model_transposed_v(QUERY_SIZE, past_capacity)
                             : build_sdpa_model(QUERY_SIZE,
                                                past_capacity,
                                                NUM_HEADS,
                                                HEAD_DIM,
                                                with_sink_and_scale,
                                                with_sink_and_scale);
    auto attention = sdpa->get_results().back()->input_value(0);
    auto transpose =
        std::make_shared<ov::op::v1::Transpose>(attention,
                                                ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    auto reshape = std::make_shared<ov::op::v1::Reshape>(
        transpose,
        ov::op::v0::Constant::create(ov::element::i64, {3}, {BATCH, QUERY_SIZE, NUM_HEADS * HEAD_DIM}),
        false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, sdpa->get_parameters());
    annotate_mask_rt_info(model, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    return model;
}

class HostFlashAttentionRuntimeTest : public ::testing::Test {
protected:
    void TearDown() override {
        if (!m_plan.empty()) {
            std::filesystem::remove(m_plan);
        }
    }

    void SetUp() override {
        m_cpu.add_extension(std::make_shared<ov::OpExtension<ov::intel_npu::op::FlashAttentionTile>>());
        m_plugin = std::make_shared<PosIdsNullPlugin>();
        m_core = std::make_shared<testing::NiceMock<ov::MockICore>>();
        m_plugin->set_core(m_core);
        ON_CALL(*m_core, get_supported_property(testing::_, testing::_, testing::_))
            .WillByDefault([](const std::string&, const ov::AnyMap& props, bool) {
                return props;
            });
        auto property = [](const std::string& name) -> ov::Any {
            if (name == ov::available_devices.name()) {
                return std::vector<std::string>{};
            }
            if (name == ov::supported_properties.name() || name == ov::internal::supported_properties.name()) {
                return std::vector<ov::PropertyName>{};
            }
            if (name == ov::intel_npu::compiler_version.name()) {
                return int64_t{0};
            }
            if (name == ov::device::architecture.name()) {
                return std::string{};
            }
            return {};
        };
        ON_CALL(*m_core, get_property(testing::_, testing::_))
            .WillByDefault([property](const std::string&, const std::string& name) {
                return property(name);
            });
        ON_CALL(*m_core, get_property(testing::_, testing::_, testing::_))
            .WillByDefault([property](const std::string&, const std::string& name, const ov::AnyMap&) {
                return property(name);
            });
        ON_CALL(*m_core,
                compile_model(testing::Matcher<const std::shared_ptr<const ov::Model>&>(testing::_),
                              testing::Matcher<const std::string&>(testing::_),
                              testing::Matcher<const ov::AnyMap&>(testing::_)))
            .WillByDefault(
                [this](const std::shared_ptr<const ov::Model>& model, const std::string&, const ov::AnyMap&) {
                    return compile_cpu(model);
                });
        ON_CALL(*m_core,
                import_model(testing::Matcher<std::istream&>(testing::_),
                             testing::Matcher<const std::string&>(testing::_),
                             testing::Matcher<const ov::AnyMap&>(testing::_)))
            .WillByDefault([this](std::istream& stream, const std::string&, const ov::AnyMap&) {
                size_t id = 0;
                ov::npuw::s11n::read(stream, id);
                auto cpu = m_cpu.import_model(stream, "CPU");
                return ov::SoPtr<ov::ICompiledModel>{
                    std::make_shared<HFACpuCompiledModel>(m_models.at(id), m_plugin, std::move(cpu), id)};
            });
    }

    ov::SoPtr<ov::ICompiledModel> compile_cpu(const std::shared_ptr<const ov::Model>& model) {
        const size_t id = m_models.size();
        m_models.push_back(model);
        auto cpu = m_cpu.compile_model(model, "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}});
        return {std::make_shared<HFACpuCompiledModel>(model, m_plugin, std::move(cpu), id)};
    }

    std::shared_ptr<ov::npuw::CompiledModel> compile(bool fused,
                                                     bool short_tiles = false,
                                                     bool alternating_masks = false,
                                                     bool with_sink_and_scale = false) {
        std::vector<std::shared_ptr<ov::Model>> models;
        ov::ParameterVector parameters;
        ov::OutputVector outputs;
        for (size_t call = 0; call < (alternating_masks ? 3u : 2u); ++call) {
            auto model = build_hfa_runtime_model(PAST_LEN, !fused, with_sink_and_scale);
            if (call != 0) {
                const auto suffix = "." + std::to_string(call);
                for (const auto& node : model->get_ops()) {
                    node->set_friendly_name(node->get_friendly_name() + suffix);
                }
                for (const auto& parameter : model->get_parameters()) {
                    parameter->output(0).set_names({parameter->output(0).get_any_name() + suffix});
                }
            }
            if (call == 2) {
                model->replace_parameter(model->get_parameters().size() - 1, models.front()->get_parameters().back());
            }
            for (const auto& parameter : model->get_parameters()) {
                if (std::find(parameters.begin(), parameters.end(), parameter) == parameters.end()) {
                    parameters.push_back(parameter);
                }
            }
            outputs.push_back(model->get_results()[0]->input_value(0));
            models.push_back(model);
        }
        m_reference_model = std::make_shared<ov::Model>(outputs, parameters);
        m_plan = std::filesystem::temp_directory_path() /
                 ("hfa_plan_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".xml");
        {
            std::ofstream plan(m_plan);
            plan << "<ensemble gflops=\"1\"><partitioning>";
            std::vector<std::vector<std::shared_ptr<ov::Node>>> grouped_layers;
            for (const auto& model : models) {
                plan << "<group gflops=\"1\" tag=\"attn\"";
                if (alternating_masks) {
                    plan << " repeated=\"hfa\"";
                }
                plan << ">";
                grouped_layers.emplace_back();
                for (const auto& node : model->get_ordered_ops()) {
                    if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Constant>(node) ||
                        ov::is_type<ov::op::v0::Result>(node)) {
                        continue;
                    }
                    grouped_layers.back().push_back(node);
                    plan << "<layer name=\"" << node->get_friendly_name() << "\"/>";
                    const auto inputs = node->input_values();
                    if (std::any_of(inputs.begin(), inputs.end(), [](const auto& input) {
                            return ov::is_type<ov::op::v0::Parameter>(input.get_node());
                        })) {
                        plan << "<input name=\"" << node->get_friendly_name() << "\"/>";
                    }
                }
                plan << "<output name=\"" << model->get_results()[0]->input_value(0).get_node()->get_friendly_name()
                     << "\"/></group>";
            }
            plan << "</partitioning>";
            if (alternating_masks) {
                plan << "<repeated><block id=\"hfa\">";
                for (size_t layer = 0; layer < grouped_layers.front().size(); ++layer) {
                    plan << "<match>";
                    for (const auto& layers : grouped_layers) {
                        OPENVINO_ASSERT(layers.size() == grouped_layers.front().size());
                        OPENVINO_ASSERT(layers[layer]->get_type_info() ==
                                        grouped_layers.front()[layer]->get_type_info());
                        plan << "<layer name=\"" << layers[layer]->get_friendly_name() << "\"/>";
                    }
                    plan << "</match>";
                }
                plan << "</block></repeated>";
            }
            plan << "</ensemble>";
            OPENVINO_ASSERT(plan.good(), "Could not write the HFA partition plan");
        }
        auto compiled =
            std::make_shared<ov::npuw::CompiledModel>(m_reference_model->clone(),
                                                      m_plugin,
                                                      ov::AnyMap{{"NPU_USE_NPUW", "YES"},
                                                                 {"NPUW_DEVICES", "CPU"},
                                                                 {"NPUW_UNFOLD_IREQS", "NO"},
                                                                 {"NPUW_FOLD", "YES"},
                                                                 {"NPUW_FUNCALL_FOR_ALL", "YES"},
                                                                 {"NPUW_FALLBACK_EXEC", "NO"},
                                                                 {"NPUW_PLAN", m_plan.string()},
                                                                 {"NPUW_ONLINE_PIPELINE", "REP"},
                                                                 {"NPUW_ONLINE_ISOLATE", "ATTN"},
                                                                 {"NPUW_ONLINE_KEEP_BLOCKS", "1"},
                                                                 {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "1"},
                                                                 {"NPUW_ONLINE_KEEP_BLOCKS_TAGGED", "attn"},
                                                                 {"NPUW_ATTN", "HFA"},
                                                                 {"NPUW_ATTN_HFA_FUSED", fused ? "YES" : "NO"},
                                                                 {"NPUW_ATTN_HFA_MASK_SKIPPING", "YES"}});
        m_hfa = nullptr;
        for (size_t idx = 0; idx < compiled->m_compiled_submodels.size(); ++idx) {
            auto& desc = compiled->m_compiled_submodels[idx];
            if (desc.replaced_by.value_or(idx) != idx) {
                continue;
            }
            if (auto* hfa = ov::npuw::attn::get_compiled_hfa(desc.pipeline.context)) {
                m_hfa = hfa;
                if (short_tiles) {
                    auto short_hfa = ov::npuw::function::HostFlashAttention::from(
                        build_hfa_runtime_model(QUERY_SIZE / 2, !fused, with_sink_and_scale),
                        fused,
                        true);
                    OPENVINO_ASSERT(short_hfa.has_value());
                    hfa->_past_tile_size = short_hfa->_past_tile_size;
                    hfa->set_compiled_tile_model(compile_cpu(short_hfa->_tile_model));
                    if (short_hfa->_masked_tile_model) {
                        hfa->set_compiled_masked_tile_model(compile_cpu(short_hfa->_masked_tile_model));
                    }
                }
            }
        }
        OPENVINO_ASSERT(m_hfa != nullptr, "Runtime fixture must contain an HFA function");
        return compiled;
    }

    void check(const std::shared_ptr<ov::npuw::CompiledModel>& compiled,
               int64_t history,
               const ov::SoPtr<ov::IAsyncInferRequest>& existing_request = {},
               bool alternating_masks = false) {
        ASSERT_NE(m_hfa, nullptr);
        auto request = existing_request ? existing_request : compiled->create_infer_request();
        bool history_set = false;
        for (const auto& state : request->query_state()) {
            if (state->get_name() == ov::npuw::HistorySizeState::name) {
                ov::Tensor tensor(ov::element::i64, {1});
                tensor.data<int64_t>()[0] = history;
                state->set_state(ov::get_tensor_impl(tensor));
                history_set = true;
            }
        }
        ASSERT_TRUE(history_set);
        auto reference =
            m_cpu.compile_model(m_reference_model, "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}})
                .create_infer_request();
        for (const auto& port : compiled->inputs()) {
            ov::Tensor tensor(port.get_element_type(), port.get_shape());
            auto* data = tensor.data<float>();
            const auto name = port.get_any_name();
            const size_t call = alternating_masks ? (name.back() == '2' ? 2 : name.back() == '1' ? 1 : 0) : 0;
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = static_cast<float>(1 + i % 7 + 3 * call) / 32.0f;
            }
            if (name.find("attention_sink.0") != std::string::npos) {
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    data[i] += static_cast<float>(history) / 32.0f;
                }
            } else if (name.find("past_key_values") != std::string::npos) {
                const bool transposed_value =
                    name.find(".value") != std::string::npos && m_hfa->_sdpa_attention_info._v_seq_dim == 3;
                for (size_t i = 0; i < tensor.get_size(); ++i) {
                    const size_t slot = (i / (transposed_value ? 1 : HEAD_DIM)) % PAST_LEN;
                    if (static_cast<int64_t>(slot) >= history) {
                        data[i] = name.find(".key") != std::string::npos ? 16.0f : 128.0f;
                    } else if (alternating_masks && slot == 1 && name.find(".value") != std::string::npos) {
                        data[i] += 2.0f + static_cast<float>(call);
                    }
                }
            } else if (name.find("mask.0") != std::string::npos) {
                const bool excludes_slot_one = alternating_masks && name == "mask.0.1";
                for (size_t q = 0; q < QUERY_SIZE; ++q) {
                    for (size_t k = 0; k < PAST_LEN + QUERY_SIZE; ++k) {
                        data[q * (PAST_LEN + QUERY_SIZE) + k] =
                            ((k < static_cast<size_t>(history) && !(excludes_slot_one && k == 1)) ||
                             (k >= PAST_LEN && k <= PAST_LEN + q))
                                ? 0.0f
                                : std::numeric_limits<float>::lowest();
                    }
                }
            }
            request->set_tensor(port, ov::get_tensor_impl(tensor));
            reference.set_tensor(name, tensor);
        }
        ASSERT_NO_THROW(request->infer());
        ASSERT_NO_THROW(reference.infer());
        for (size_t output = 0; output < compiled->outputs().size(); ++output) {
            auto actual = request->get_tensor(compiled->outputs()[output]);
            auto expected = reference.get_output_tensor(output);
            ASSERT_EQ(actual->get_shape(), expected.get_shape());
            for (size_t i = 0; i < expected.get_size(); ++i) {
                ASSERT_NEAR(actual->data<float>()[i], expected.data<float>()[i], 1e-4f)
                    << "output " << output << " element " << i;
            }
        }
    }

    size_t count(const ov::SoPtr<ov::ICompiledModel>& model) {
        return model ? std::static_pointer_cast<HFACpuCompiledModel>(model._ptr)->infer_count.load() : 0;
    }

    ov::Core m_cpu;
    std::shared_ptr<PosIdsNullPlugin> m_plugin;
    std::shared_ptr<testing::NiceMock<ov::MockICore>> m_core;
    std::vector<std::shared_ptr<const ov::Model>> m_models;
    std::shared_ptr<ov::Model> m_reference_model;
    std::filesystem::path m_plan;
    ov::npuw::compiled::HostFlashAttention* m_hfa = nullptr;
};

TEST_F(HostFlashAttentionRuntimeTest, FusedPartialHistory) {
    auto compiled = compile(true);
    ASSERT_NE(m_hfa, nullptr);
    ASSERT_TRUE(m_hfa->_compiled_masked_tile_model);
    EXPECT_EQ(m_hfa->_compiled_masked_tile_model->inputs().size(), m_hfa->_compiled_tile_model->inputs().size() + 1);
    check(compiled, QUERY_SIZE + 3);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), count(m_hfa->_compiled_final_tile_model));
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), count(m_hfa->_compiled_final_tile_model));
}

TEST_F(HostFlashAttentionRuntimeTest, FusedPartialHistoryWithSinkAndScale) {
    auto compiled = compile(true, false, false, true);
    ASSERT_NE(m_hfa, nullptr);
    ASSERT_TRUE(m_hfa->_compiled_masked_tile_model);
    auto request = compiled->create_infer_request();
    check(compiled, QUERY_SIZE + 3, request);
    check(compiled, 0, request);
    check(compiled, QUERY_SIZE + 7, request);
    EXPECT_GT(count(m_hfa->_compiled_masked_tile_model), 0u);
}

TEST_F(HostFlashAttentionRuntimeTest, FusedShortTilesPartialHistorySkipsUpperTile) {
    auto compiled = compile(true, true);
    check(compiled, QUERY_SIZE / 2 + 3);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), count(m_hfa->_compiled_final_tile_model));
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), count(m_hfa->_compiled_final_tile_model));
}

TEST_F(HostFlashAttentionRuntimeTest, FusedShortTilesSkipsUpperTile) {
    auto compiled = compile(true, true);
    check(compiled, 3);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), 0u);
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), count(m_hfa->_compiled_final_tile_model));
}

TEST_F(HostFlashAttentionRuntimeTest, FusedAlignedHistoryDoesNotUseMaskedTile) {
    auto compiled = compile(true);
    check(compiled, PAST_LEN);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), (PAST_LEN / QUERY_SIZE) * count(m_hfa->_compiled_final_tile_model));
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), 0u);
}

TEST_F(HostFlashAttentionRuntimeTest, FusedEmptyHistoryWithPastCapacity) {
    auto compiled = compile(true);
    auto request = compiled->create_infer_request();
    check(compiled, 0, request);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), 0u);
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), 0u);
    check(compiled, QUERY_SIZE + 3, request);
    check(compiled, 0, request);
}

TEST_F(HostFlashAttentionRuntimeTest, NonFusedEmptyHistoryWithPastCapacity) {
    auto compiled = compile(false);
    auto request = compiled->create_infer_request();
    check(compiled, 0, request);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), 0u);
    check(compiled, QUERY_SIZE + 3, request);
    check(compiled, 0, request);
}

TEST_F(HostFlashAttentionRuntimeTest, NonFusedPartialHistorySkipsUpperTile) {
    auto compiled = compile(false, true);
    check(compiled, 3);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), count(m_hfa->_compiled_final_tile_model));
    EXPECT_FALSE(m_hfa->_compiled_masked_tile_model);
}

TEST_F(HostFlashAttentionRuntimeTest, FusedFoldedCallsWithAlternatingMasks) {
    auto compiled = compile(true, false, true);
    ASSERT_EQ(compiled->outputs().size(), 3u);
    ASSERT_EQ(compiled->m_compiled_submodels.size(), 3u);
    for (const auto& desc : compiled->m_compiled_submodels) {
        ASSERT_TRUE(desc.replaced_by.has_value());
        ASSERT_EQ(*desc.replaced_by, 0u);
    }
    ASSERT_NE(m_hfa, nullptr);
    ASSERT_TRUE(m_hfa->_compiled_masked_tile_model);
    check(compiled, 3, {}, true);
    EXPECT_EQ(count(m_hfa->_compiled_tile_model), 0u);
    EXPECT_EQ(count(m_hfa->_compiled_masked_tile_model), 3u);
    EXPECT_EQ(count(m_hfa->_compiled_final_tile_model), 3u);
}

TEST_F(HostFlashAttentionRuntimeTest, MaskedTileExportImport) {
    auto compiled = compile(true);
    std::stringstream stream;
    ASSERT_NO_THROW(compiled->export_model(stream));
    auto imported = ov::npuw::CompiledModel::import_model(stream, m_plugin, {});
    m_hfa = nullptr;
    for (auto& desc : imported->m_compiled_submodels) {
        if (auto* hfa = ov::npuw::attn::get_compiled_hfa(desc.pipeline.context)) {
            m_hfa = hfa;
        }
    }
    ASSERT_NE(m_hfa, nullptr);
    ASSERT_TRUE(m_hfa->_compiled_masked_tile_model);
    check(imported, QUERY_SIZE + 3);
}

TEST_F(HostFlashAttentionRuntimeTest, MaskedTileWithSinkAndScaleExportImport) {
    auto compiled = compile(true, false, false, true);
    std::stringstream stream;
    ASSERT_NO_THROW(compiled->export_model(stream));
    auto imported = ov::npuw::CompiledModel::import_model(stream, m_plugin, {});
    m_hfa = nullptr;
    for (auto& desc : imported->m_compiled_submodels) {
        if (auto* hfa = ov::npuw::attn::get_compiled_hfa(desc.pipeline.context)) {
            m_hfa = hfa;
        }
    }
    ASSERT_NE(m_hfa, nullptr);
    ASSERT_TRUE(m_hfa->_compiled_masked_tile_model);
    check(imported, QUERY_SIZE + 3);
}

}  // namespace
