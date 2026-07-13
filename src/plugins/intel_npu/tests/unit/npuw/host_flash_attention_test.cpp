// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_flash_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
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
                                            size_t head_dim = HEAD_DIM) {
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

TEST(HostFlashAttentionFromTest, Fused_MaskTileAtIndexSixInFinalTileOnly) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_tile_model->inputs().size(), 6u);
    EXPECT_EQ(result->_final_tile_model->inputs().size(), 7u);
    expect_input_name(result->_final_tile_model, 6, "MASK_TILE", "fused final tile");
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
    EXPECT_EQ(result->_tile_size, static_cast<int64_t>(QUERY_SIZE));
    EXPECT_EQ(result->_query_size, QUERY_SIZE);
}

TEST(HostFlashAttentionFromTest, Fused_ContextSizeIsCorrect) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->_context_size, QUERY_SIZE + PAST_LEN);
}

// ============================================================================
// Input / output shape checks
// Expected shapes (BATCH=1, NUM_HEADS=8, HEAD_DIM=64, QUERY_SIZE=16):
//   past_acc  [1, 8, 16, 64]   past_max  [1, 8, 16, 1]   past_d  [1, 8, 16, 1]
//   k_tile    [1, 8, 16, 64]   v_tile    [1, 8, 64, 16]  (V stored pre-transposed)
//   q         [1, 8, 16, 64]   mask_tile [1, 1, 16, 16]
//   regular tile outputs: acc [1,8,16,64]  maxx [1,8,16,1]  d [1,8,16,1]
//   final tile output:    [1, QUERY_SIZE, NUM_HEADS*HEAD_DIM] = [1, 16, 512]
// ============================================================================

// Fused path — regular tile (6 inputs, no mask)
TEST(HostFlashAttentionFromTest, Fused_RegularTileInputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile  [B, kv_heads, tile, head_dim]
        {BATCH, NUM_HEADS, HEAD_DIM, QUERY_SIZE},  // v_tile  [B, kv_heads, head_dim, tile]
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
    };
    check_input_shapes(result->_tile_model, expected_inputs, "fused regular tile");
}

// Fused path — final tile (7 inputs, with mask)
TEST(HostFlashAttentionFromTest, Fused_FinalTileInputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), true);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile
        {BATCH, NUM_HEADS, HEAD_DIM, QUERY_SIZE},  // v_tile
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
        {BATCH, 1, QUERY_SIZE, QUERY_SIZE},        // mask_tile [B, 1, seq, tile]
    };
    check_input_shapes(result->_final_tile_model, expected_inputs, "fused final tile");
}

// Non-fused path — regular tile (7 inputs, with mask)
TEST(HostFlashAttentionFromTest, NonFused_RegularTileInputShapes) {
    auto result = ov::npuw::function::HostFlashAttention::from(build_sdpa_model(), false);
    ASSERT_TRUE(result.has_value());
    // [past_acc, past_max, past_d, k_tile, v_tile, q, mask_tile]
    const std::vector<ov::Shape> expected_inputs = {
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // past_acc
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_max
        {BATCH, NUM_HEADS, QUERY_SIZE, 1},         // past_d
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // k_tile  [B, kv_heads, tile, head_dim]
        {BATCH, NUM_HEADS, HEAD_DIM, QUERY_SIZE},  // v_tile  [B, kv_heads, head_dim, tile]
        {BATCH, NUM_HEADS, QUERY_SIZE, HEAD_DIM},  // q
        {BATCH, 1, QUERY_SIZE, QUERY_SIZE},        // mask_tile [B, 1, seq, tile]
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
