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
 * - ExpertBatchMode: Tests transformation for batch mode (N=8 total experts, K=4 active, token=1)
 * - ExpertIterativeMode: Tests transformation for iterative mode (N=8 total experts, K=1 active, token=16, chunk=8)
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

// Create MoE expert graph with full GPT-OSS pattern including dual branches
// Pattern matches GPTOSSExpert in moe.cpp:
// Tile -> Reshape -> MatMul (gate+up) -> Add -> Slice (activation) -> Minimum -> Swish
//                                           \-> Slice (gate) -> Clamp -> Add2
// Then: Swish + Add2 -> Multiply1 -> MatMul (down) -> Add3 -> Reshape -> Multiply (output)
std::shared_ptr<Model> create_gpt_oss_expert_graph(size_t num_experts,
                                                   size_t hidden_dim = 2880,
                                                   size_t token_count = 1,
                                                   bool with_awq_multiply = false) {
    ov::ParameterVector params;

    // 1. Expert input: [token_count, hidden_dim]
    auto expert_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{token_count, hidden_dim});
    expert_input->set_friendly_name("expert_input");
    params.push_back(expert_input);

    // 2. Tile to replicate for experts
    auto repeats =
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1});
    auto tile = std::make_shared<op::v0::Tile>(expert_input, repeats);
    tile->set_friendly_name("expert_tile");

    // 3. Reshape to 3D: [num_experts, token_count, hidden_dim]
    auto reshape_shape1 = op::v0::Constant::create(element::i64,
                                                   Shape{3},
                                                   std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                        static_cast<int64_t>(token_count),
                                                                        static_cast<int64_t>(hidden_dim)});
    auto reshape1 = std::make_shared<op::v1::Reshape>(tile, reshape_shape1, false);
    reshape1->set_friendly_name("expert_reshape1");

    // 4. First MatMul (gate + up projections) with quantized weights
    // Weight parameter: [num_experts, hidden_dim*2, hidden_dim] (NF4)
    auto weights_nf4_1 =
        std::make_shared<op::v0::Parameter>(element::nf4, Shape{num_experts, hidden_dim * 2, hidden_dim});
    weights_nf4_1->set_friendly_name("weights_nf4_1");
    params.push_back(weights_nf4_1);

    // Convert NF4 to FP16
    auto weights_fp16_1 = std::make_shared<op::v0::Convert>(weights_nf4_1, element::f16);
    weights_fp16_1->set_friendly_name("weights_convert_fp16_1");

    // Scale parameter: [num_experts, hidden_dim*2, 1] (FP16)
    auto weights_scale_1 = std::make_shared<op::v0::Parameter>(element::f16, Shape{num_experts, hidden_dim * 2, 1});
    weights_scale_1->set_friendly_name("weights_scale_1");
    params.push_back(weights_scale_1);

    // Multiply with scale (dequantization)
    auto weights_mult1 = std::make_shared<op::v1::Multiply>(weights_fp16_1, weights_scale_1);
    weights_mult1->set_friendly_name("weights_multiply1");

    // Convert FP16 to FP32 for MatMul
    auto weights_conv1 = std::make_shared<op::v0::Convert>(weights_mult1, element::f32);
    weights_conv1->set_friendly_name("weights_convert1");

    auto matmul1 = std::make_shared<op::v0::MatMul>(reshape1, weights_conv1, false, true);
    matmul1->set_friendly_name("matmul1");

    // Biases: [num_experts, 1, hidden_dim*2]
    auto biases1 = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, 1, hidden_dim * 2});
    biases1->set_friendly_name("biases1");
    params.push_back(biases1);

    auto add1 = std::make_shared<op::v1::Add>(matmul1, biases1);
    add1->set_friendly_name("add1");

    // 5. Dual branches from Add1
    // Activation branch: Slice -> Minimum -> Swish [-> AWQ Multiply if enabled]
    auto slice_start1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto slice_stop1 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim)});
    auto slice_step1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto slice_axis1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto slice1 = std::make_shared<op::v8::Slice>(add1, slice_start1, slice_stop1, slice_step1, slice_axis1);
    slice1->set_friendly_name("slice_activation");

    auto minimum_const = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{20.0f});
    auto minimum = std::make_shared<op::v1::Minimum>(slice1, minimum_const);
    minimum->set_friendly_name("minimum");

    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.0f});
    auto swish = std::make_shared<op::v4::Swish>(minimum, swish_beta);
    swish->set_friendly_name("swish");

    // Optional AWQ multiply after Swish
    std::shared_ptr<ov::Node> activation_output = swish;
    if (with_awq_multiply) {
        auto awq_scale = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, token_count, hidden_dim});
        awq_scale->set_friendly_name("awq_activation_scale");
        params.push_back(awq_scale);

        auto awq_mult = std::make_shared<op::v1::Multiply>(swish, awq_scale);
        awq_mult->set_friendly_name("awq_activation_multiply");
        activation_output = awq_mult;
    }

    // Gate branch: Slice -> Clamp -> Add2
    auto slice_start2 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim)});
    auto slice_stop2 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim * 2)});
    auto slice_step2 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto slice_axis2 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto slice2 = std::make_shared<op::v8::Slice>(add1, slice_start2, slice_stop2, slice_step2, slice_axis2);
    slice2->set_friendly_name("slice_gate");

    auto clamp = std::make_shared<op::v0::Clamp>(slice2, -20.0f, 20.0f);
    clamp->set_friendly_name("clamp");

    auto add2_const = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{0.0f});
    auto add2 = std::make_shared<op::v1::Add>(clamp, add2_const);
    add2->set_friendly_name("add2");

    // 6. Merge branches: Multiply1
    auto multiply1 = std::make_shared<op::v1::Multiply>(activation_output, add2);
    multiply1->set_friendly_name("multiply1_merge");

    // 7. Second MatMul (down projection) with quantized weights
    // Weight parameter: [num_experts, hidden_dim, hidden_dim] (NF4)
    auto weights_nf4_2 = std::make_shared<op::v0::Parameter>(element::nf4, Shape{num_experts, hidden_dim, hidden_dim});
    weights_nf4_2->set_friendly_name("weights_nf4_2");
    params.push_back(weights_nf4_2);

    // Convert NF4 to FP16
    auto weights_fp16_2 = std::make_shared<op::v0::Convert>(weights_nf4_2, element::f16);
    weights_fp16_2->set_friendly_name("weights_convert_fp16_2");

    // Scale parameter: [num_experts, hidden_dim, 1] (FP16)
    auto weights_scale_2 = std::make_shared<op::v0::Parameter>(element::f16, Shape{num_experts, hidden_dim, 1});
    weights_scale_2->set_friendly_name("weights_scale_2");
    params.push_back(weights_scale_2);

    // Multiply with scale (dequantization)
    auto weights_mult2 = std::make_shared<op::v1::Multiply>(weights_fp16_2, weights_scale_2);
    weights_mult2->set_friendly_name("weights_multiply2");

    // Convert FP16 to FP32 for MatMul
    auto weights_conv2 = std::make_shared<op::v0::Convert>(weights_mult2, element::f32);
    weights_conv2->set_friendly_name("weights_convert2");

    auto matmul2 = std::make_shared<op::v0::MatMul>(multiply1, weights_conv2, false, true);
    matmul2->set_friendly_name("matmul2");

    // Biases: [num_experts, 1, hidden_dim]
    auto biases2 = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, 1, hidden_dim});
    biases2->set_friendly_name("biases2");
    params.push_back(biases2);

    auto add3 = std::make_shared<op::v1::Add>(matmul2, biases2);
    add3->set_friendly_name("add3");

    // 8. Output reshape
    auto reshape_shape2 = op::v0::Constant::create(element::i64,
                                                   Shape{3},
                                                   std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                        static_cast<int64_t>(token_count),
                                                                        static_cast<int64_t>(hidden_dim)});
    auto reshape2 = std::make_shared<op::v1::Reshape>(add3, reshape_shape2, false);
    reshape2->set_friendly_name("expert_reshape2");

    // 9. Router scores multiply
    auto router_scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_experts, token_count, 1});
    router_scores->set_friendly_name("router_scores");
    params.push_back(router_scores);

    auto output_multiply = std::make_shared<op::v1::Multiply>(reshape2, router_scores);
    output_multiply->set_friendly_name("output_multiply");

    // 10. Result
    auto result = std::make_shared<op::v0::Result>(output_multiply);
    result->set_friendly_name("output");

    auto model = std::make_shared<Model>(ResultVector{result}, params);
    model->set_friendly_name("gpt_oss_expert_" + std::to_string(num_experts) + (with_awq_multiply ? "_awq" : ""));

    return model;
}

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

// Test expert batch mode: N total experts, K active experts (K < N), token_count = 1
TEST_F(MoETransformationTest, ExpertBatchMode) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 4;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 1;  // EXPERT_BATCH: single token

    auto model = create_transformable_moe_graph(num_total_experts, hidden_dim, token_count);
    save_model(model, "moe_expert_batch_before");

    auto structure_info = analyze_moe_structure(model);
    ASSERT_TRUE(structure_info.has_value()) << "Failed to analyze MoE structure";
    EXPECT_EQ(structure_info->num_experts, num_total_experts);
    EXPECT_EQ(structure_info->input_token_count, token_count);
    EXPECT_TRUE(structure_info->is_expert_batch_mode());

    // Transform to K active experts for batch mode
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = 0;

    MoEModelTransformer transformer(*structure_info);
    auto transformed = transformer.apply_expert_transformation(model, config);

    ASSERT_NE(transformed, nullptr) << "Expert batch mode transformation failed";
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    save_model(transformed, "moe_expert_batch_after");

    // Expert batch mode with K>1: expect unrolling, more nodes created
    EXPECT_GT(transformed->get_ordered_ops().size(), model->get_ordered_ops().size())
        << "Expected node count increase after unrolling " << num_active_experts << " experts";

    // After unrolling, parameters are duplicated for each active expert
    EXPECT_GT(transformed->get_parameters().size(), model->get_parameters().size())
        << "Expected parameter count increase after unrolling";

    // Result count should remain the same
    EXPECT_EQ(transformed->get_results().size(), model->get_results().size());

    // Verify output shape: [num_active_experts, token_count, hidden_dim*2]
    // After batch mode transformation, expert dimension should be num_active_experts and token dimension should be 1
    auto result = transformed->get_results()[0];
    auto output_shape = result->get_input_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static()) << "Output shape should be static";

    auto shape = output_shape.to_shape();
    EXPECT_EQ(shape.size(), 3) << "Output should be 3D tensor";
    EXPECT_EQ(shape[0], num_active_experts)
        << "Expert dimension should be " << num_active_experts << " after batch mode transformation";
    EXPECT_EQ(shape[1], token_count) << "Token dimension should be 1 for batch mode";
    EXPECT_EQ(shape[2], hidden_dim * 2) << "Hidden dimension should remain unchanged";
}

// Test expert iterative mode: N total experts, 1 active expert, token_count > 1
TEST_F(MoETransformationTest, ExpertIterativeMode) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 1;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 16;  // EXPERT_ITERATIVE: multiple tokens
    constexpr size_t chunk_size = 8;

    auto model = create_transformable_moe_graph(num_total_experts, hidden_dim, token_count);
    save_model(model, "moe_expert_iterative_before");

    auto structure_info = analyze_moe_structure(model);
    ASSERT_TRUE(structure_info.has_value()) << "Failed to analyze MoE structure";
    EXPECT_EQ(structure_info->num_experts, num_total_experts);
    EXPECT_EQ(structure_info->input_token_count, token_count);
    EXPECT_FALSE(structure_info->is_expert_batch_mode());

    // Transform to single expert with chunking for iterative mode
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = chunk_size;

    MoEModelTransformer transformer(*structure_info);
    auto transformed = transformer.apply_expert_transformation(model, config);

    ASSERT_NE(transformed, nullptr) << "Expert iterative mode transformation failed";
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    save_model(transformed, "moe_expert_iterative_after");

    // Iterative mode with single expert: no unrolling happens (num_target_experts=1)
    // Tile is replaced with Reshape, token count is adjusted to chunk size
    // Parameters count preserved (no unrolling for single expert)
    EXPECT_EQ(transformed->get_parameters().size(), model->get_parameters().size())
        << "Expected parameter count to stay the same (no unrolling for single expert)";

    // Result count should remain the same
    EXPECT_EQ(transformed->get_results().size(), model->get_results().size());

    // Verify output shape: [num_target_experts, chunk_size, hidden_dim*2]
    // After iterative mode transformation, expert dimension should be 1 and token dimension should be chunk_size
    auto result = transformed->get_results()[0];
    auto output_shape = result->get_input_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static()) << "Output shape should be static";

    auto shape = output_shape.to_shape();
    EXPECT_EQ(shape.size(), 3) << "Output should be 3D tensor";
    EXPECT_EQ(shape[0], num_active_experts) << "Expert dimension should be 1 after iterative mode transformation";
    EXPECT_EQ(shape[1], chunk_size) << "Token dimension should be chunk_size after iterative mode transformation";
    EXPECT_EQ(shape[2], hidden_dim * 2) << "Hidden dimension should remain unchanged";
}

// Test AWQ multiply unrolling: Verify that AWQ quantization multiply nodes are correctly unrolled
TEST_F(MoETransformationTest, AWQMultiplyUnrolling) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 4;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 1;
    constexpr bool with_awq = true;

    // Create GPT-OSS expert graph with AWQ multiply after Swish
    auto model = create_gpt_oss_expert_graph(num_total_experts, hidden_dim, token_count, with_awq);
    save_model(model, "moe_expert_awq_before");

    auto structure_info = analyze_moe_structure(model);
    ASSERT_TRUE(structure_info.has_value()) << "Failed to analyze MoE structure with AWQ multiply";
    EXPECT_EQ(structure_info->num_experts, num_total_experts);
    EXPECT_EQ(structure_info->input_token_count, token_count);
    EXPECT_TRUE(structure_info->is_expert_batch_mode());

    // Transform to K active experts
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = 0;

    MoEModelTransformer transformer(*structure_info);
    auto transformed = transformer.apply_expert_transformation(model, config);

    ASSERT_NE(transformed, nullptr) << "AWQ multiply unrolling transformation failed";
    EXPECT_NO_THROW(transformed->validate_nodes_and_infer_types());

    save_model(transformed, "moe_expert_awq_after");

    // Verify that AWQ multiply nodes were unrolled
    size_t awq_multiply_count = 0;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (auto mult = std::dynamic_pointer_cast<op::v1::Multiply>(node)) {
            std::string name = mult->get_friendly_name();
            if (name.find("awq_activation_multiply") != std::string::npos) {
                awq_multiply_count++;
            }
        }
    }

    // Should have num_active_experts AWQ multiply nodes after unrolling
    EXPECT_EQ(awq_multiply_count, num_active_experts)
        << "Expected " << num_active_experts << " AWQ multiply nodes after unrolling, found " << awq_multiply_count;

    // Verify AWQ scale parameters were unrolled
    size_t awq_param_count = 0;
    for (const auto& param : transformed->get_parameters()) {
        std::string name = param->get_friendly_name();
        if (name.find("awq_activation_scale") != std::string::npos) {
            awq_param_count++;
        }
    }

    // Should have num_active_experts AWQ scale parameters after unrolling
    EXPECT_EQ(awq_param_count, num_active_experts)
        << "Expected " << num_active_experts << " AWQ scale parameters after unrolling, found " << awq_param_count;

    // Verify output shape is correct
    auto result = transformed->get_results()[0];
    auto output_shape = result->get_input_partial_shape(0);
    ASSERT_TRUE(output_shape.is_static()) << "Output shape should be static";

    auto shape = output_shape.to_shape();
    EXPECT_EQ(shape.size(), 3) << "Output should be 3D tensor";
    EXPECT_EQ(shape[0], num_active_experts) << "Expert dimension should be " << num_active_experts;
    EXPECT_EQ(shape[1], token_count) << "Token dimension should be 1";
    EXPECT_EQ(shape[2], hidden_dim) << "Hidden dimension should remain unchanged";
}

// Test comparison: Verify AWQ and non-AWQ models have similar structure after unrolling
TEST_F(MoETransformationTest, AWQvsNonAWQComparison) {
    constexpr size_t num_total_experts = 8;
    constexpr size_t num_active_experts = 2;
    constexpr size_t hidden_dim = 2880;
    constexpr size_t token_count = 1;

    // Create both AWQ and non-AWQ models
    auto model_awq = create_gpt_oss_expert_graph(num_total_experts, hidden_dim, token_count, true);
    auto model_non_awq = create_gpt_oss_expert_graph(num_total_experts, hidden_dim, token_count, false);

    // Transform both
    MoETransformConfig config;
    config.num_target_experts = num_active_experts;
    config.chunk_size = 0;

    auto structure_awq = analyze_moe_structure(model_awq);
    auto structure_non_awq = analyze_moe_structure(model_non_awq);

    ASSERT_TRUE(structure_awq.has_value());
    ASSERT_TRUE(structure_non_awq.has_value());

    MoEModelTransformer transformer_awq(*structure_awq);
    MoEModelTransformer transformer_non_awq(*structure_non_awq);

    auto transformed_awq = transformer_awq.apply_expert_transformation(model_awq, config);
    auto transformed_non_awq = transformer_non_awq.apply_expert_transformation(model_non_awq, config);

    ASSERT_NE(transformed_awq, nullptr);
    ASSERT_NE(transformed_non_awq, nullptr);

    // AWQ model should have more operations due to additional multiply nodes
    EXPECT_GT(transformed_awq->get_ordered_ops().size(), transformed_non_awq->get_ordered_ops().size())
        << "AWQ model should have more operations (includes AWQ multiply nodes)";

    // AWQ model should have more parameters due to AWQ scale parameters
    EXPECT_GT(transformed_awq->get_parameters().size(), transformed_non_awq->get_parameters().size())
        << "AWQ model should have more parameters (includes AWQ scale parameters)";

    // Both should have same number of results
    EXPECT_EQ(transformed_awq->get_results().size(), transformed_non_awq->get_results().size());

    // Both should have same output shape
    auto result_awq = transformed_awq->get_results()[0];
    auto result_non_awq = transformed_non_awq->get_results()[0];

    EXPECT_EQ(result_awq->get_input_partial_shape(0), result_non_awq->get_input_partial_shape(0))
        << "AWQ and non-AWQ models should have identical output shapes";
}

}  // namespace
