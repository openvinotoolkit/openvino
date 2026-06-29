// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyramid_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"

namespace {

// Build a minimal single-layer decomposed SDPA model with contiguous KV cache:
//   query, past_key, past_value, new_key, new_value, mask  ->
//   Concat(past_key, new_key) -> MatMul(query, key_concat^T) -> Add(+mask)
//   -> Softmax -> MatMul(softmax, value_concat)
std::shared_ptr<ov::Model> build_isolated_attention_model(size_t query_len, size_t past_len) {
    using namespace ov;

    const size_t num_heads = 4;
    const size_t head_dim = 16;
    const size_t context_len = past_len + query_len;

    const Shape past_shape = {1, num_heads, past_len, head_dim};
    const Shape new_token_shape = {1, num_heads, query_len, head_dim};
    const Shape mask_shape = {1, 1, query_len, context_len};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    auto query = make_param("query.0", new_token_shape);
    auto past_key = make_param("past_key_values.0.key", past_shape);
    auto past_value = make_param("past_key_values.0.value", past_shape);
    auto new_key = make_param("new_key.0", new_token_shape);
    auto new_value = make_param("new_value.0", new_token_shape);
    auto mask = make_param("mask.0", mask_shape);

    auto key_concat = std::make_shared<op::v0::Concat>(OutputVector{past_key, new_key}, 2);
    key_concat->set_friendly_name("concat_key.0");
    auto value_concat = std::make_shared<op::v0::Concat>(OutputVector{past_value, new_value}, 2);
    value_concat->set_friendly_name("concat_value.0");

    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    add->set_friendly_name("add.0");
    auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
    softmax->set_friendly_name("softmax.0");
    auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_concat->output(0));
    matmul2->set_friendly_name("matmul2.0");

    auto make_result = [&](const Output<Node>& out, const std::string& name) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(name);
        results.push_back(r);
    };

    make_result(key_concat->output(0), "present.0.key");
    make_result(value_concat->output(0), "present.0.value");
    make_result(matmul2->output(0), "attn_out.0");

    auto model = std::make_shared<Model>(results, params, "isolated_attention_model");
    model->validate_nodes_and_infer_types();
    return model;
}

// Build a single-layer decomposed SDPA model with block-split KV cache:
//   past_key_values.0.key_block_0 .. key_block_{N-1}, new_key -> Concat (axis 2)
//   past_key_values.0.value_block_0 .. value_block_{N-1}, new_value -> Concat (axis 2)
//   Then standard MatMul -> Add -> Softmax -> MatMul structure.
std::shared_ptr<ov::Model> build_block_kv_attention_model(size_t num_blocks, size_t block_size = 4) {
    using namespace ov;

    const size_t num_heads = 4;
    const size_t head_dim = 16;
    const size_t total_kv_len = num_blocks * block_size + 1;

    const Shape block_shape = {1, num_heads, block_size, head_dim};
    const Shape new_token_shape = {1, num_heads, 1, head_dim};
    const Shape mask_shape = {1, 1, 1, total_kv_len};

    ParameterVector params;
    ResultVector results;

    auto make_param = [&](const std::string& name, const Shape& shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        p->set_friendly_name(name);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    OutputVector key_inputs, value_inputs;
    for (size_t b = 0; b < num_blocks; ++b) {
        const std::string bsuf = "_block_" + std::to_string(b);
        key_inputs.push_back(make_param("past_key_values.0.key" + bsuf, block_shape)->output(0));
        value_inputs.push_back(make_param("past_key_values.0.value" + bsuf, block_shape)->output(0));
    }

    auto query = make_param("query.0", new_token_shape);
    auto new_key = make_param("new_key.0", new_token_shape);
    auto new_value = make_param("new_value.0", new_token_shape);
    auto mask = make_param("mask.0", mask_shape);

    key_inputs.push_back(new_key->output(0));
    value_inputs.push_back(new_value->output(0));

    auto key_concat = std::make_shared<op::v0::Concat>(key_inputs, 2);
    key_concat->set_friendly_name("concat_key.0");
    auto value_concat = std::make_shared<op::v0::Concat>(value_inputs, 2);
    value_concat->set_friendly_name("concat_value.0");

    auto qk = std::make_shared<op::v0::MatMul>(query, key_concat, false, true);
    qk->set_friendly_name("matmul1.0");
    auto add = std::make_shared<op::v1::Add>(qk->output(0), mask->output(0));
    add->set_friendly_name("add.0");
    auto softmax = std::make_shared<op::v8::Softmax>(add->output(0), 3);
    softmax->set_friendly_name("softmax.0");
    auto matmul2 = std::make_shared<op::v0::MatMul>(softmax->output(0), value_concat->output(0));
    matmul2->set_friendly_name("matmul2.0");

    auto make_result = [&](const Output<Node>& out, const std::string& name) {
        auto r = std::make_shared<op::v0::Result>(out);
        r->set_friendly_name(name);
        results.push_back(r);
    };

    make_result(key_concat->output(0), "present.0.key");
    make_result(value_concat->output(0), "present.0.value");
    make_result(matmul2->output(0), "attn_out.0");

    auto model = std::make_shared<Model>(results, params, "block_kv_attention_model");
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

// ---- validate_and_setup_pyramid_attention: contiguous path ----

TEST(PyramidAttentionTest, ValidateReturnsContiguousResultForStandardModel) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<ov::npuw::function::PyramidValidationContiguousResult>(*result));
}

TEST(PyramidAttentionTest, ValidateContiguousResultHasCorrectQueryAndContextLengths) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& cont = std::get<ov::npuw::function::PyramidValidationContiguousResult>(*result);
    EXPECT_EQ(cont.query_length, 1u);
    EXPECT_EQ(cont.full_context_length, 2048u);
    EXPECT_EQ(cont.past_kv_length, 2047u);
    EXPECT_TRUE(cont.is_valid());
}

TEST(PyramidAttentionTest, ValidateContiguousResultHasSequenceDimsForKvParams) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/63);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& cont = std::get<ov::npuw::function::PyramidValidationContiguousResult>(*result);
    ASSERT_FALSE(cont.past_key_sequence_dims.empty());
    ASSERT_FALSE(cont.past_value_sequence_dims.empty());
    // Concat is along axis 2 (sequence dimension)
    EXPECT_EQ(cont.past_key_sequence_dims.at("past_key_values.0.key"), 2u);
    EXPECT_EQ(cont.past_value_sequence_dims.at("past_key_values.0.value"), 2u);
}

// ---- validate_and_setup_pyramid_attention: block path ----

TEST(PyramidAttentionTest, ValidateReturnsBlockResultForBlockKvCacheModel) {
    auto model = build_block_kv_attention_model(/*num_blocks=*/3);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(std::holds_alternative<ov::npuw::function::PyramidValidationBlockResult>(*result));
}

TEST(PyramidAttentionTest, ValidateBlockResultHasNonEmptyBlockIndices) {
    // 3 past key blocks + 3 past value blocks
    auto model = build_block_kv_attention_model(/*num_blocks=*/3);

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    ASSERT_TRUE(result.has_value());
    const auto& blk = std::get<ov::npuw::function::PyramidValidationBlockResult>(*result);
    EXPECT_EQ(blk.past_key_block_global_param_indices.size(), 3u);
    EXPECT_EQ(blk.past_value_block_global_param_indices.size(), 3u);
    EXPECT_TRUE(blk.is_valid());
}

// ---- validate_and_setup_pyramid_attention: negative case ----

TEST(PyramidAttentionTest, ValidateReturnsNulloptForModelWithoutSdpaPattern) {
    // A trivial model with no MatMul+Softmax pattern should return nullopt
    using namespace ov;
    auto in = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});
    in->set_friendly_name("input");
    auto out = std::make_shared<op::v0::Result>(in);
    auto model = std::make_shared<Model>(ResultVector{out}, ParameterVector{in}, "trivial_model");

    auto result = ov::npuw::function::validate_and_setup_pyramid_attention(model);

    EXPECT_FALSE(result.has_value());
}

// ---- PyramidAttention::from (function namespace) ----

TEST(PyramidAttentionTest, FromSucceedsOnValidGenerateModel) {
    // generate mode: query_len=1, context=2048 -> step=1024 -> 2 pyramid models
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    EXPECT_TRUE(pyramid->is_valid());
    EXPECT_EQ(pyramid->_query_length, 1u);
    EXPECT_EQ(pyramid->_full_context_length, 2048u);
}

TEST(PyramidAttentionTest, FromCreatesExpectedNumberOfModels) {
    // context=2048, pyramid_step=1024 -> num_models = 2048/1024 = 2
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    EXPECT_EQ(pyramid->num_models(), 2u);
    EXPECT_EQ(pyramid->_models.size(), pyramid->_attentions.size());
}

TEST(PyramidAttentionTest, FromReusesOriginalModelForLastPyramidEntry) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);

    auto pyramid = ov::npuw::function::PyramidAttention::from(model);

    ASSERT_TRUE(pyramid.has_value());
    EXPECT_EQ(pyramid->_models.back().get(), model.get());
}

// ---- compiled::PyramidAttention::make (abstract base + subclass construction) ----

TEST(PyramidAttentionTest, CompiledMakeReturnsContiguousSubclassForNonBlockInput) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);
    auto func_pyramid = ov::npuw::function::PyramidAttention::from(model);
    ASSERT_TRUE(func_pyramid.has_value());

    auto compiled = ov::npuw::compiled::PyramidAttention::make(*func_pyramid);

    ASSERT_NE(compiled, nullptr);
    EXPECT_FALSE(compiled->is_block_mode());
    EXPECT_NE(dynamic_cast<ov::npuw::compiled::PyramidAttentionContiguous*>(compiled.get()), nullptr);
    EXPECT_EQ(dynamic_cast<ov::npuw::compiled::PyramidAttentionBlock*>(compiled.get()), nullptr);
}

TEST(PyramidAttentionTest, CompiledMakePopulatesNumModelsFromFunctionPyramid) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);
    auto func_pyramid = ov::npuw::function::PyramidAttention::from(model);
    ASSERT_TRUE(func_pyramid.has_value());

    auto compiled = ov::npuw::compiled::PyramidAttention::make(*func_pyramid);

    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->num_models(), func_pyramid->num_models());
}

TEST(PyramidAttentionTest, CompiledContiguousBlockPortStubsReturnEmpty) {
    auto model = build_isolated_attention_model(/*query_len=*/1, /*past_len=*/2047);
    auto func_pyramid = ov::npuw::function::PyramidAttention::from(model);
    ASSERT_TRUE(func_pyramid.has_value());

    auto compiled = ov::npuw::compiled::PyramidAttention::make(*func_pyramid);

    ASSERT_NE(compiled, nullptr);
    for (size_t i = 0; i < compiled->num_models(); ++i) {
        EXPECT_TRUE(compiled->key_block_port_set_at(i).empty());
        EXPECT_TRUE(compiled->val_block_port_set_at(i).empty());
        EXPECT_TRUE(compiled->key_block_port_map_at(i).empty());
        EXPECT_TRUE(compiled->val_block_port_map_at(i).empty());
    }
    EXPECT_EQ(compiled->num_key_blocks_global(), 0u);
}

TEST(PyramidAttentionTest, CompiledContiguousQuerySizeMatchesInputQueryLen) {
    const size_t query_len = 1;
    auto model = build_isolated_attention_model(query_len, /*past_len=*/2047);
    auto func_pyramid = ov::npuw::function::PyramidAttention::from(model);
    ASSERT_TRUE(func_pyramid.has_value());

    auto compiled = ov::npuw::compiled::PyramidAttention::make(*func_pyramid);

    ASSERT_NE(compiled, nullptr);
    for (size_t i = 0; i < compiled->num_models(); ++i) {
        EXPECT_EQ(compiled->query_size_at(i), query_len);
    }
}
