// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_transformations/moe_transformation.hpp"

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "openvino/op/ops.hpp"
#include "openvino/pass/serialize.hpp"

/*
 * Test suite for MoE Expert Transformation
 *
 * Testing Strategy:
 * - DecodingMode: Tests transformation for decoding (N=8 total experts, K=4 active, token=1)
 * - PrefillMode: Tests transformation for prefill (N=8 total experts, K=1 active, token=16, chunk=8)
 */

// Uncomment to save debug XML files during test execution
// #define SAVE_TEST_MODELS

namespace {

using namespace ov;
using namespace ov::npuw::function;

// ============================================================================
// Test Utilities
// ============================================================================

class MoETransformationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for all tests
    }

    void TearDown() override {
        // Common cleanup
    }

    // Helper: Save model to XML for debugging
    void save_model(const std::shared_ptr<Model>& model, const std::string& prefix) {
#ifdef SAVE_TEST_MODELS
        std::string xml_path = prefix + ".xml";
        std::string bin_path = prefix + ".bin";
        ov::pass::Serialize serialize_pass(xml_path, bin_path);
        serialize_pass.run_on_model(const_cast<std::shared_ptr<Model>&>(model));
#endif
    }
};

// ============================================================================
// Synthetic Test Graph Builders
// ============================================================================

// Create MoE graph with complete structure required by apply_expert_transformation
// Pattern: Parameter -> Tile -> Reshape -> MatMul -> Add -> Reshape -> Multiply -> Result
std::shared_ptr<Model> create_transformable_moe_graph(size_t num_experts,
                                                      size_t hidden_dim = 2880,
                                                      size_t token_count = 1) {
    ov::ParameterVector params;

    // 1. Expert input: [token_count, hidden_dim]
    auto expert_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{token_count, hidden_dim});
    expert_input->set_friendly_name("expert_input");
    params.push_back(expert_input);

    // 2. Tile to replicate for experts: [token_count, hidden_dim] -> [num_experts*token_count, hidden_dim]
    auto repeats =
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1});
    auto tile = std::make_shared<op::v0::Tile>(expert_input, repeats);
    tile->set_friendly_name("expert_tile");

    // 3. Reshape to 3D: [num_experts, token_count, hidden_dim]
    auto reshape_shape = op::v0::Constant::create(element::i64,
                                                  Shape{3},
                                                  std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                       static_cast<int64_t>(token_count),
                                                                       static_cast<int64_t>(hidden_dim)});
    auto reshape_in = std::make_shared<op::v1::Reshape>(tile, reshape_shape, false);
    reshape_in->set_friendly_name("expert_reshape_in");

    // 4. Expert weights with quantization pattern (NF4 -> FP16 -> Multiply with scale -> FP32)
    // Weight parameter: [num_experts, hidden_dim*2, hidden_dim] (NF4)
    auto weights_nf4 =
        std::make_shared<op::v0::Parameter>(element::nf4, Shape{num_experts, hidden_dim * 2, hidden_dim});
    weights_nf4->set_friendly_name("expert_weights_nf4");
    params.push_back(weights_nf4);

    // Convert NF4 to FP16
    auto weights_fp16 = std::make_shared<op::v0::Convert>(weights_nf4, element::f16);
    weights_fp16->set_friendly_name("expert_weights_convert_fp16");

    // Scale parameter: [num_experts, hidden_dim*2, 1] (FP16)
    auto weights_scale = std::make_shared<op::v0::Parameter>(element::f16, Shape{num_experts, hidden_dim * 2, 1});
    weights_scale->set_friendly_name("expert_weights_scale");
    params.push_back(weights_scale);

    // Multiply with scale (dequantization)
    auto weights_scaled = std::make_shared<op::v1::Multiply>(weights_fp16, weights_scale);
    weights_scaled->set_friendly_name("expert_weights_scaled");

    // Convert FP16 to FP32 for MatMul
    auto weights_fp32 = std::make_shared<op::v0::Convert>(weights_scaled, element::f32);
    weights_fp32->set_friendly_name("expert_weights_convert_fp32");

    // 5. MatMul
    auto matmul = std::make_shared<op::v0::MatMul>(reshape_in, weights_fp32, false, true);  // transpose_b=true
    matmul->set_friendly_name("expert_matmul");

    // 6. Biases: [num_experts, 1, hidden_dim*2]
    auto biases = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, 1, hidden_dim * 2});
    biases->set_friendly_name("expert_biases");
    params.push_back(biases);

    // 7. Add bias
    auto add = std::make_shared<op::v1::Add>(matmul, biases);
    add->set_friendly_name("expert_add");

    // 8. Reshape output: [num_experts, token_count, hidden_dim*2]
    auto reshape_out_shape = op::v0::Constant::create(element::i64,
                                                      Shape{3},
                                                      std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                           static_cast<int64_t>(token_count),
                                                                           static_cast<int64_t>(hidden_dim * 2)});
    auto reshape_out = std::make_shared<op::v1::Reshape>(add, reshape_out_shape, false);
    reshape_out->set_friendly_name("expert_reshape_out");

    // 9. Router scores: [num_experts, token_count, 1]
    auto router_scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, token_count, 1});
    router_scores->set_friendly_name("router_scores");
    params.push_back(router_scores);

    // 10. Multiply with router scores (REQUIRED by apply_expert_transformation)
    auto multiply = std::make_shared<op::v1::Multiply>(reshape_out, router_scores);
    multiply->set_friendly_name("router_multiply");

    // 11. Result
    auto result = std::make_shared<op::v0::Result>(multiply);
    result->set_friendly_name("output");

    auto model = std::make_shared<Model>(ResultVector{result}, params);
    model->set_friendly_name("moe_" + std::to_string(num_experts) + "_experts");

    return model;
}

// ============================================================================
// Unit Tests - MoE Expert Transformation
// ============================================================================

// Test decoding mode: N total experts, K active experts (K < N), token_count = 1
TEST_F(MoETransformationTest, DecodingMode) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 4;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 1;  // Decoding: single token

    auto model = create_transformable_moe_graph(num_total_experts, hidden_dim, token_count);
    save_model(model, "moe_decoding_before");

    auto structure_info = analyze_moe_structure(model);
    ASSERT_TRUE(structure_info.has_value()) << "Failed to analyze MoE structure";
    EXPECT_EQ(structure_info->num_experts, num_total_experts);
    EXPECT_EQ(structure_info->input_token_count, token_count);
    EXPECT_TRUE(structure_info->is_decoding_stage);

    // Transform to K active experts for decoding
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = 0;

    MoEModelTransformer transformer(*structure_info);
    auto transformed = transformer.apply_expert_transformation(model, config);

    ASSERT_NE(transformed, nullptr) << "Decoding transformation failed";
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    save_model(transformed, "moe_decoding_after");

    // Decoding with K>1: expect unrolling, more nodes created
    EXPECT_GT(transformed->get_ordered_ops().size(), model->get_ordered_ops().size())
        << "Expected node count increase after unrolling " << num_active_experts << " experts";

    // After unrolling, parameters are duplicated for each active expert
    EXPECT_GT(transformed->get_parameters().size(), model->get_parameters().size())
        << "Expected parameter count increase after unrolling";

    // Result count should remain the same
    EXPECT_EQ(transformed->get_results().size(), model->get_results().size());

    // Verify output shape: [num_active_experts, token_count, hidden_dim*2]
    // After decoding transformation, expert dimension should be num_active_experts and token dimension should be 1
    auto result = transformed->get_results()[0];
    auto output_shape = result->get_input_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static()) << "Output shape should be static";

    auto shape = output_shape.to_shape();
    EXPECT_EQ(shape.size(), 3) << "Output should be 3D tensor";
    EXPECT_EQ(shape[0], num_active_experts)
        << "Expert dimension should be " << num_active_experts << " after decoding transformation";
    EXPECT_EQ(shape[1], token_count) << "Token dimension should be 1 for decoding";
    EXPECT_EQ(shape[2], hidden_dim * 2) << "Hidden dimension should remain unchanged";
}

// Test prefill mode: N total experts, 1 active expert, token_count > 1
TEST_F(MoETransformationTest, PrefillMode) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 1;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 16;  // Prefill: multiple tokens
    constexpr size_t chunk_size = 8;

    auto model = create_transformable_moe_graph(num_total_experts, hidden_dim, token_count);
    save_model(model, "moe_prefill_before");

    auto structure_info = analyze_moe_structure(model);
    ASSERT_TRUE(structure_info.has_value()) << "Failed to analyze MoE structure";
    EXPECT_EQ(structure_info->num_experts, num_total_experts);
    EXPECT_EQ(structure_info->input_token_count, token_count);
    EXPECT_FALSE(structure_info->is_decoding_stage);

    // Transform to single expert with chunking for prefill
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = chunk_size;

    MoEModelTransformer transformer(*structure_info);
    auto transformed = transformer.apply_expert_transformation(model, config);

    ASSERT_NE(transformed, nullptr) << "Prefill transformation failed";
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    save_model(transformed, "moe_prefill_after");

    // Prefill with single expert: no unrolling happens (num_target_experts=1)
    // Tile is replaced with Reshape, token count is adjusted via fix_token_count_for_prefill
    // Parameters count preserved (no unrolling for single expert)
    EXPECT_EQ(transformed->get_parameters().size(), model->get_parameters().size())
        << "Expected parameter count to stay the same (no unrolling for single expert)";

    // Result count should remain the same
    EXPECT_EQ(transformed->get_results().size(), model->get_results().size());

    // Verify output shape: [num_target_experts, chunk_size, hidden_dim*2]
    // After prefill transformation, expert dimension should be 1 and token dimension should be chunk_size
    auto result = transformed->get_results()[0];
    auto output_shape = result->get_input_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static()) << "Output shape should be static";

    auto shape = output_shape.to_shape();
    EXPECT_EQ(shape.size(), 3) << "Output should be 3D tensor";
    EXPECT_EQ(shape[0], num_active_experts) << "Expert dimension should be 1 after prefill transformation";
    EXPECT_EQ(shape[1], chunk_size) << "Token dimension should be chunk_size after prefill transformation";
    EXPECT_EQ(shape[2], hidden_dim * 2) << "Hidden dimension should remain unchanged";
}

}  // namespace
