// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/openvino.hpp"
#include "npuw_transformations/convert_kvcache_to_precision.hpp"
#include "npuw_transformations/split_kvcache_into_blocks.hpp"
#include "pyramid_attention.hpp"
#include "util.hpp"

namespace {

// Build a single-layer isolated attention subgraph model suitable for pyramid attention.
// This mimics the attention block after NPUW_ONLINE_ISOLATE=ATTN extraction.
//
// Graph structure:
//   query [1, num_heads, query_len, head_dim]
//   past_key [1, num_heads, past_len, head_dim]  --> Concat(axis=2) with new_key --> key_cache
//   past_value [1, num_heads, past_len, head_dim] --> Concat(axis=2) with new_value --> value_cache
//   new_key [1, num_heads, query_len, head_dim]
//   new_value [1, num_heads, query_len, head_dim]
//   mask [1, 1, query_len, context_len]
//
//   MatMul(query, key_cache^T) -> Add(mask) -> Softmax -> MatMul(value_cache) -> attn_out
//
struct AttentionModelConfig {
    size_t num_heads = 4;
    size_t head_dim = 16;
    size_t query_len = 1;
    size_t past_len = 63;
    size_t num_layers = 1;
};

std::shared_ptr<ov::Model> build_isolated_attention_model(const AttentionModelConfig& cfg) {
    using namespace ov;

    const size_t context_len = cfg.past_len + cfg.query_len;
    const Shape past_shape = {1, cfg.num_heads, cfg.past_len, cfg.head_dim};
    const Shape new_token_shape = {1, cfg.num_heads, cfg.query_len, cfg.head_dim};
    const Shape mask_shape = {1, 1, cfg.query_len, context_len};

    ParameterVector params;
    ResultVector results;

    for (size_t n = 0; n < cfg.num_layers; ++n) {
        const std::string idx = std::to_string(n);

        auto make_param = [&](const std::string& name, const Shape& shape) {
            auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
            p->set_friendly_name(name);
            p->output(0).get_tensor().set_names({name});
            params.push_back(p);
            return p;
        };

        auto query = make_param("query." + idx, new_token_shape);
        auto past_key = make_param("past_key_values." + idx + ".key", past_shape);
        auto past_value = make_param("past_key_values." + idx + ".value", past_shape);
        auto new_key = make_param("new_key." + idx, new_token_shape);
        auto new_value = make_param("new_value." + idx, new_token_shape);
        auto mask = make_param("mask." + idx, mask_shape);

        auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
        key_concat->set_friendly_name("concat_key." + idx);

        auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{past_value, new_value}, 2);
        value_concat->set_friendly_name("concat_value." + idx);

        auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
        qk->set_friendly_name("matmul1." + idx);

        auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
        add->set_friendly_name("add." + idx);

        auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
        softmax->set_friendly_name("softmax." + idx);

        auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_concat->output(0));
        matmul2->set_friendly_name("matmul2." + idx);

        auto make_result = [&](const ov::Output<ov::Node>& out, const std::string& name) {
            auto r = std::make_shared<op::v0::Result>(out);
            r->set_friendly_name(name);
            r->output(0).get_tensor().set_names({name});
            results.push_back(r);
        };

        make_result(key_concat->output(0), "present." + idx + ".key");
        make_result(value_concat->output(0), "present." + idx + ".value");
        make_result(matmul2->output(0), "attn_out." + idx);
    }

    auto model = std::make_shared<Model>(results, params, "isolated_attention_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Helper function to extract contiguous result from validation variant
const ov::npuw::function::PyramidValidationContiguousResult& get_contiguous_result(
    const ov::npuw::function::PyramidValidationResult& validation) {
    return std::get<ov::npuw::function::PyramidValidationContiguousResult>(validation);
}

// Helper function to extract block result from validation variant
const ov::npuw::function::PyramidValidationBlockResult& get_block_result(
    const ov::npuw::function::PyramidValidationResult& validation) {
    return std::get<ov::npuw::function::PyramidValidationBlockResult>(validation);
}
// Helper function to apply SplitKVCacheIntoBlocks transformation
std::shared_ptr<ov::Model> apply_split_kvcache_into_blocks(
    const std::shared_ptr<ov::Model>& model,
    uint32_t block_size = 32) {
    auto cloned = model->clone();
    // Use v_transposed=false to match test model structure where both key and value use axis 2
    ov::npuw::pass::SplitKVCacheIntoBlocks(block_size, false).run_on_model(cloned);
    return cloned;
}
// --- Tests for validate_and_setup_pyramid_attention (Contiguous KV Cache) ---

TEST(PyramidAttentionTest, ValidateSucceedsOnValidAttentionModel) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(result.has_value());
    
    const auto& contiguous = get_contiguous_result(*result);
    EXPECT_TRUE(contiguous.is_valid());
    EXPECT_EQ(contiguous.query_length, 1u);
    EXPECT_EQ(contiguous.full_context_length, 64u);
    EXPECT_EQ(contiguous.past_kv_length, 63u);
    EXPECT_FALSE(contiguous.past_key_sequence_dims.empty());
    EXPECT_FALSE(contiguous.past_value_sequence_dims.empty());
}

TEST(PyramidAttentionTest, ValidateExtractsCorrectSequenceDimForSingleLayer) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 31;
    auto model = build_isolated_attention_model(cfg);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value()) << "Validation failed for single-layer model";
    const auto& contiguous = get_contiguous_result(*result);
    
    // Concat axis is 2 (sequence dim)
    auto key_it = contiguous.past_key_sequence_dims.find("past_key_values.0.key");
    EXPECT_NE(key_it, contiguous.past_key_sequence_dims.end()) 
        << "Key 'past_key_values.0.key' not found. Map has " << contiguous.past_key_sequence_dims.size() << " entries.";
    if (key_it != contiguous.past_key_sequence_dims.end()) {
        EXPECT_EQ(key_it->second, 2u);
    }
    
    auto val_it = contiguous.past_value_sequence_dims.find("past_key_values.0.value");
    EXPECT_NE(val_it, contiguous.past_value_sequence_dims.end())
        << "Key 'past_key_values.0.value' not found. Map has " << contiguous.past_value_sequence_dims.size() << " entries.";
    if (val_it != contiguous.past_value_sequence_dims.end()) {
        EXPECT_EQ(val_it->second, 2u);
    }
}

TEST(PyramidAttentionTest, DebugRegexPatternMatching) {
    // Verify that our regex patterns match expected parameter names
    EXPECT_TRUE(ov::npuw::util::isPastKeyValuesKeyContiguous("past_key_values.0.key").has_value());
    EXPECT_EQ(ov::npuw::util::isPastKeyValuesKeyContiguous("past_key_values.0.key").value(), 0);
    
    EXPECT_TRUE(ov::npuw::util::isPastKeyValuesKeyContiguous("past_key_values.1.key").has_value());
    EXPECT_EQ(ov::npuw::util::isPastKeyValuesKeyContiguous("past_key_values.1.key").value(), 1);
    
    EXPECT_TRUE(ov::npuw::util::isPastKeyValuesValueContiguous("past_key_values.0.value").has_value());
    EXPECT_EQ(ov::npuw::util::isPastKeyValuesValueContiguous("past_key_values.0.value").value(), 0);
    
    EXPECT_TRUE(ov::npuw::util::isPastKeyValuesValueContiguous("past_key_values.1.value").has_value());
    EXPECT_EQ(ov::npuw::util::isPastKeyValuesValueContiguous("past_key_values.1.value").value(), 1);
    
    // Should NOT match
    EXPECT_FALSE(ov::npuw::util::isPastKeyValuesKeyContiguous("query.0").has_value());
    EXPECT_FALSE(ov::npuw::util::isPastKeyValuesKeyContiguous("past_key_values.0.key_block_0").has_value());
}

TEST(PyramidAttentionTest, DebugMultiLayerModelParameters) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    EXPECT_EQ(model->get_parameters().size(), 6u);  // 1 layer * 6 params per layer

    for (const auto& param : model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        SCOPED_TRACE("Parameter: " + name);
    }
}

TEST(PyramidAttentionTest, ValidateExtractsCorrectDimsForMultipleLayers) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& contiguous = get_contiguous_result(*result);

    EXPECT_EQ(contiguous.past_key_sequence_dims.size(), 1u);
    EXPECT_EQ(contiguous.past_value_sequence_dims.size(), 1u);

    auto key_it = contiguous.past_key_sequence_dims.find("past_key_values.0.key");
    EXPECT_NE(key_it, contiguous.past_key_sequence_dims.end());
    if (key_it != contiguous.past_key_sequence_dims.end()) {
        EXPECT_EQ(key_it->second, 2u);
    }

    auto val_it = contiguous.past_value_sequence_dims.find("past_key_values.0.value");
    EXPECT_NE(val_it, contiguous.past_value_sequence_dims.end());
    if (val_it != contiguous.past_value_sequence_dims.end()) {
        EXPECT_EQ(val_it->second, 2u);
    }
}

TEST(PyramidAttentionTest, ValidateSucceedsForPrefillChunkModel) {
    AttentionModelConfig cfg;
    cfg.query_len = 128;
    cfg.past_len = 128;  // past_len = one chunk already processed
    auto model = build_isolated_attention_model(cfg);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& contiguous = get_contiguous_result(*result);
    EXPECT_EQ(contiguous.query_length, 128u);
    EXPECT_EQ(contiguous.full_context_length, 256u);
    EXPECT_EQ(contiguous.past_kv_length, 128u);
}

// --- Tests for process_pyramid_model ---

TEST(PyramidAttentionTest, ProcessPyramidModelSucceedsForGenerateCase) {
    // Generate case: query_len=1, past_len=63, context=64
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    // Process pyramid model for model_idx=0 (smallest pyramid)
    // pyramid_step=1024 for generate, but full_context=64, so use step that fits
    const size_t pyramid_step = 32;
    const size_t model_idx = 0;

    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            model_idx,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->is_valid());
    EXPECT_NE(result->model, nullptr);

    // The cloned model should have been reshaped
    // For generate: current_context = (0+1)*32 + (64-1-63) = 32
    //              current_past = 32 - 1 = 31
    // Verify past_key parameter was reshaped to past_len=31
    for (const auto& param : result->model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        if (ov::npuw::util::isPastKeyValuesKey(name) || ov::npuw::util::isPastKeyValuesValue(name)) {
            EXPECT_EQ(param->get_shape()[2], 31u) << "Parameter " << name << " sequence dim should be 31";
        }
    }
}

TEST(PyramidAttentionTest, ProcessPyramidModelSucceedsForPrefillCase) {
    // Prefill case: query_len=128, past_len=128, context=256
    AttentionModelConfig cfg;
    cfg.query_len = 128;
    cfg.past_len = 128;
    auto model = build_isolated_attention_model(cfg);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    // For prefill: pyramid_step = query_length = 128
    const size_t pyramid_step = 128;
    const size_t model_idx = 0;

    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            model_idx,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->is_valid());

    // For prefill: current_context = (0+1)*128 = 128
    //             current_past = 128 - 128 = 0
    // Past KV params should be reshaped to past_len=0
    for (const auto& param : result->model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        if (ov::npuw::util::isPastKeyValuesKey(name) || ov::npuw::util::isPastKeyValuesValue(name)) {
            EXPECT_EQ(param->get_shape()[2], 0u) << "Parameter " << name << " sequence dim should be 0";
        }
    }
}

TEST(PyramidAttentionTest, ProcessPyramidModelProducesCorrectAttentionInfo) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    const size_t pyramid_step = 32;
    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            0,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());

    // Attention should have mask and KV inputs
    EXPECT_NE(result->attention._mask, nullptr);
    EXPECT_FALSE(result->attention._inputs.empty());
    // Each layer contributes 1 key + 1 value param = 2 inputs for 1 layer
    EXPECT_EQ(result->attention._inputs.size(), 2u);

    // Mask shape should reflect new context length
    // For generate: current_context = 32 + 0 = 32
    EXPECT_EQ(result->attention._mask_shape[3], 32u);
}

TEST(PyramidAttentionTest, ProcessPyramidModelClonesAndDoesNotModifyOriginal) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    // Record original shapes
    std::map<std::string, ov::Shape> original_shapes;
    for (const auto& param : model->get_parameters()) {
        original_shapes[param->get_friendly_name()] = param->get_shape();
    }

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            0,
                                                            32,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());

    // Original model shapes must be unchanged
    for (const auto& param : model->get_parameters()) {
        EXPECT_EQ(param->get_shape(), original_shapes.at(param->get_friendly_name()))
            << "Original model parameter " << param->get_friendly_name() << " was modified";
    }
}

TEST(PyramidAttentionTest, ProcessMultiplePyramidModelsProducesGrowingContextLengths) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 127;  // context = 128, step = 32 -> 4 models
    auto model = build_isolated_attention_model(cfg);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    const size_t pyramid_step = 32;
    const size_t num_models = contiguous.full_context_length / pyramid_step;
    ASSERT_EQ(num_models, 4u);

    std::vector<size_t> context_lengths;
    for (size_t i = 0; i < num_models - 1; ++i) {
        auto result = ov::npuw::function::process_pyramid_model(model,
                                                                i,
                                                                pyramid_step,
                                                                contiguous.query_length,
                                                                contiguous.past_kv_length,
                                                                contiguous.full_context_length,
                                                                contiguous.past_key_sequence_dims,
                                                                contiguous.past_value_sequence_dims);
        ASSERT_TRUE(result.has_value()) << "Failed to process pyramid model " << i;
        context_lengths.push_back(result->attention.context_len());
    }
    // Last model uses original
    context_lengths.push_back(contiguous.full_context_length);

    // Context lengths should be strictly increasing
    for (size_t i = 1; i < context_lengths.size(); ++i) {
        EXPECT_GT(context_lengths[i], context_lengths[i - 1])
            << "Context length at index " << i << " should be greater than at " << (i - 1);
    }
}

// --- Tests for PyramidAttention::from ---

TEST(PyramidAttentionTest, FromSucceedsOnValidGenerateModel) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;  // context = 2048, step = 1024 -> 2 models
    auto model = build_isolated_attention_model(cfg);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    EXPECT_TRUE(pyramid->is_valid());
    EXPECT_EQ(pyramid->_query_length, 1u);
    EXPECT_EQ(pyramid->_full_context_length, 2048u);
    EXPECT_EQ(pyramid->num_models(), 2u);
    EXPECT_EQ(pyramid->_models.size(), pyramid->_attentions.size());
}

TEST(PyramidAttentionTest, FromSucceedsOnValidPrefillModel) {
    AttentionModelConfig cfg;
    cfg.query_len = 128;
    cfg.past_len = 128;  // context = 256, step = 128 -> 2 models
    auto model = build_isolated_attention_model(cfg);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    EXPECT_TRUE(pyramid->is_valid());
    EXPECT_EQ(pyramid->_query_length, 128u);
    EXPECT_EQ(pyramid->_full_context_length, 256u);
    EXPECT_EQ(pyramid->num_models(), 2u);
}

TEST(PyramidAttentionTest, FromReusesOriginalModelForLastPyramidModel) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    // The last model should be the original model pointer (optimization)
    EXPECT_EQ(pyramid->_models.back().get(), model.get());
}

// --- Tests for pyramid attention after KV cache compression (i8) ---

TEST(PyramidAttentionTest, ConvertKVCacheToI8AddsExtraParameters) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    const size_t params_before = model->get_parameters().size();

    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    // After i8 conversion, extra scale/zp parameters are added for dequantization
    EXPECT_GT(model->get_parameters().size(), params_before)
        << "ConvertKVCacheToPrecision(i8) should add dequantization parameters";

    // Verify the past_key parameter is now i8
    bool found_i8_key = false;
    for (const auto& param : model->get_parameters()) {
        if (ov::npuw::util::isPastKeyValuesKey(param->get_friendly_name())) {
            EXPECT_EQ(param->get_element_type(), ov::element::i8);
            found_i8_key = true;
        }
    }
    EXPECT_TRUE(found_i8_key);
}

TEST(PyramidAttentionTest, ValidateSucceedsAfterI8KVCacheConversion) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    // Apply i8 KV cache compression - inserts DQ nodes between past_kv and Concat
    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    // validate_and_setup_pyramid_attention should still find the SDPA pattern
    // because the Concat node is still present (DQ nodes are before Concat, not after)
    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value()) << "SDPA pattern should be detectable after i8 KV cache conversion";
    const auto& contiguous = get_contiguous_result(*result);
    EXPECT_TRUE(contiguous.is_valid());
    EXPECT_EQ(contiguous.query_length, 1u);
    EXPECT_EQ(contiguous.full_context_length, 2048u);
    EXPECT_FALSE(contiguous.past_key_sequence_dims.empty());
    EXPECT_FALSE(contiguous.past_value_sequence_dims.empty());
}

TEST(PyramidAttentionTest, ProcessPyramidModelAfterI8KVCacheConversion) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    // Apply i8 KV cache compression
    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    const size_t pyramid_step = 1024;
    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            0,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value())
        << "process_pyramid_model should succeed after i8 KV cache conversion";
}

TEST(PyramidAttentionTest, ProcessPyramidModelI8IncludesDQParamsInAttentionInputs) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    const size_t pyramid_step = 1024;
    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            0,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());

    // Per layer: 2 KV params (key + value)
    //          + 3 DQ params (key_scale, key_zp for asymmetric key; value_scale for symmetric value)
    //          = 5 attention inputs
    EXPECT_EQ(result->attention._inputs.size(), 5u)
        << "Attention inputs must include DQ scale/zp parameters so pyramid runtime can view/slice them";
}

TEST(PyramidAttentionTest, ProcessPyramidModelI8ReshapesDQParamsCorrectly) {
    AttentionModelConfig cfg;
    cfg.query_len = 128;
    cfg.past_len = 128;
    auto model = build_isolated_attention_model(cfg);

    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(model);
    ASSERT_TRUE(validation.has_value());
    const auto& contiguous = get_contiguous_result(*validation);

    // Prefill pyramid_00: current_context = 128, current_past = 0
    const size_t pyramid_step = 128;
    auto result = ov::npuw::function::process_pyramid_model(model,
                                                            0,
                                                            pyramid_step,
                                                            contiguous.query_length,
                                                            contiguous.past_kv_length,
                                                            contiguous.full_context_length,
                                                            contiguous.past_key_sequence_dims,
                                                            contiguous.past_value_sequence_dims);

    ASSERT_TRUE(result.has_value());

    // Verify DQ params were reshaped to match pyramid level's past_len=0
    for (const auto& param : result->model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        if (ov::npuw::util::isDQScaleOrZPKey(name)) {
            // Key DQ params: seq dim is 2 → should be 0 for first pyramid level
            EXPECT_EQ(param->get_shape()[2], 0u) << "DQ key param " << name << " seq dim should be 0";
        } else if (ov::npuw::util::isDQScaleOrZPValue(name)) {
            // Value DQ params: seq dim is 2 → should be 0 for first pyramid level
            EXPECT_EQ(param->get_shape()[2], 0u) << "DQ value param " << name << " seq dim should be 0";
        }
    }
}

TEST(PyramidAttentionTest, FromI8ModelIncludesDQParamsInCompiledAttentionInfos) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 2047;
    auto model = build_isolated_attention_model(cfg);

    ov::npuw::ConvertKVCacheToPrecision(ov::element::i8).run_on_model(model);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);
    ASSERT_TRUE(pyramid.has_value());
    EXPECT_EQ(pyramid->num_models(), 2u);

    // Each model's attention should have 5 inputs (2 KV + 3 DQ)
    for (size_t i = 0; i < pyramid->num_models(); ++i) {
        EXPECT_EQ(pyramid->_attentions[i]._inputs.size(), 5u)
            << "Pyramid model " << i << " attention must include DQ params";
    }
}

// --- Tests for validate_and_setup_pyramid_attention (Block-Split KV Cache) ---
// Block-split mode uses block indices instead of contiguous sequence dimensions.
// This mode is used when KV cache is split into independent blocks.

TEST(PyramidAttentionTest, DebugBlockModeTransformation) {
    // Debug test to verify SplitKVCacheIntoBlocks transformation is applied
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    auto block_model = apply_split_kvcache_into_blocks(model, 32);

    // Verify that block parameters exist
    bool found_block_0 = false, found_block_tail = false;
    size_t block_params = 0;
    for (const auto& param : block_model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        if (name.find("_block_") != std::string::npos) {
            block_params++;
            if (name.find("_block_0") != std::string::npos) found_block_0 = true;
            if (name.find("_block_tail") != std::string::npos) found_block_tail = true;
        }
    }
    
    EXPECT_TRUE(found_block_0) << "Block 0 parameters not found after split";
    EXPECT_TRUE(found_block_tail) << "Block tail parameters not found after split";
    EXPECT_GE(block_params, 4u) << "Expected at least 4 block params (2 keys + 2 values), got " << block_params;
    
    // Verify that original contiguous parameters were removed
    for (const auto& param : block_model->get_parameters()) {
        const auto& name = param->get_friendly_name();
        EXPECT_NE(name, "past_key_values.0.key") << "Original key parameter should be removed";
        EXPECT_NE(name, "past_key_values.0.value") << "Original value parameter should be removed";
    }
}

TEST(PyramidAttentionTest, ValidateSucceedsOnBlockModeModel) {
    // Build model with block-structured KV cache
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation to model
    auto block_model = apply_split_kvcache_into_blocks(model, 32);

    // After block split, model should validate and return PyramidValidationBlockResult
    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    
    ASSERT_TRUE(result.has_value());
    const auto& block_result = get_block_result(*result);
    EXPECT_TRUE(block_result.is_valid());
}

TEST(PyramidAttentionTest, ValidateExtractsCorrectBlockIndicesForSingleLayer) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 31;
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation
    auto block_model = apply_split_kvcache_into_blocks(model, 32);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    ASSERT_TRUE(result.has_value());
    const auto& block_result = get_block_result(*result);
    EXPECT_EQ(block_result.past_key_block_global_param_indices.size(), 1u);
    EXPECT_EQ(block_result.past_value_block_global_param_indices.size(), 1u);
}

TEST(PyramidAttentionTest, ProcessPyramidModelSucceedsForBlockModeGenerateCase) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation
    auto block_model = apply_split_kvcache_into_blocks(model, 32);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    ASSERT_TRUE(validation.has_value());
    const auto& block_result = get_block_result(*validation);

    const size_t pyramid_step = 32;
    auto result = ov::npuw::function::process_pyramid_model(block_model,
                                                            0,
                                                            pyramid_step,
                                                            block_result.query_length,
                                                            0,  // full_past_kv_length not used in block mode
                                                            block_result.full_context_length,
                                                            {},  // past_key_sequence_dims not used
                                                            {},  // past_value_sequence_dims not used
                                                            true);  // is_block_split = true

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->is_valid());
    EXPECT_NE(result->model, nullptr);
}

TEST(PyramidAttentionTest, ProcessPyramidModelSucceedsForBlockModePrefillCase) {
    AttentionModelConfig cfg;
    cfg.query_len = 128;
    cfg.past_len = 128;
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation
    auto block_model = apply_split_kvcache_into_blocks(model, 128);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    ASSERT_TRUE(validation.has_value());
    const auto& block_result = get_block_result(*validation);

    const size_t pyramid_step = 128;
    auto result = ov::npuw::function::process_pyramid_model(block_model,
                                                            0,
                                                            pyramid_step,
                                                            block_result.query_length,
                                                            0,
                                                            block_result.full_context_length,
                                                            {},
                                                            {},
                                                            true);

    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->is_valid());
}

TEST(PyramidAttentionTest, ProcessBlockModeModelClonesAndDoesNotModifyOriginal) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 63;
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation and record original state
    auto block_model = apply_split_kvcache_into_blocks(model, 32);
    std::vector<std::string> original_param_names;
    for (const auto& param : block_model->get_parameters()) {
        original_param_names.push_back(param->get_friendly_name());
    }

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    ASSERT_TRUE(validation.has_value());
    const auto& block_result = get_block_result(*validation);

    auto result = ov::npuw::function::process_pyramid_model(block_model,
                                                            0,
                                                            32,
                                                            block_result.query_length,
                                                            0,
                                                            block_result.full_context_length,
                                                            {},
                                                            {},
                                                            true);

    ASSERT_TRUE(result.has_value());
    // Original model parameters must not be modified by processing
    std::vector<std::string> current_param_names;
    for (const auto& param : block_model->get_parameters()) {
        current_param_names.push_back(param->get_friendly_name());
    }
    EXPECT_EQ(original_param_names, current_param_names);
}

TEST(PyramidAttentionTest, ProcessBlockModePyramidModelsProducesGrowingContextLengths) {
    AttentionModelConfig cfg;
    cfg.query_len = 1;
    cfg.past_len = 127;  // context = 128, step = 32 -> 4 models
    auto model = build_isolated_attention_model(cfg);

    // Apply SplitKVCacheIntoBlocks transformation
    auto block_model = apply_split_kvcache_into_blocks(model, 32);

    auto validation = ov::npuw::function::validate_and_setup_pyramid_attention(block_model);
    ASSERT_TRUE(validation.has_value());
    const auto& block_result = get_block_result(*validation);

    const size_t pyramid_step = 32;
    const size_t num_models = block_result.full_context_length / pyramid_step;
    ASSERT_EQ(num_models, 4u);

    std::vector<size_t> context_lengths;
    for (size_t i = 0; i < num_models - 1; ++i) {
        auto result = ov::npuw::function::process_pyramid_model(block_model,
                                                                i,
                                                                pyramid_step,
                                                                block_result.query_length,
                                                                0,
                                                                block_result.full_context_length,
                                                                {},
                                                                {},
                                                                true);
        ASSERT_TRUE(result.has_value()) << "Failed to process pyramid model " << i;
        context_lengths.push_back(result->attention.context_len());
    }
    // Last model uses original
    context_lengths.push_back(block_result.full_context_length);

    // Context lengths should be strictly increasing
    for (size_t i = 1; i < context_lengths.size(); ++i) {
        EXPECT_GT(context_lengths[i], context_lengths[i - 1])
            << "Context length at index " << i << " should be greater than at " << (i - 1);
    }
}

}  // namespace
