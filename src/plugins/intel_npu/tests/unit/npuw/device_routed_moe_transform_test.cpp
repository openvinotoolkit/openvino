// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_transformations/device_routed_moe_transform.hpp"

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

/*
 * Test suite for Device-Routed MoE Transformation
 *
 * GPT-OSS tests (create_complete_moe_graph):
 * - BasicTransformation:       Gather insertion in quantized weights; shape updates
 * - MultiLayerMoE:             Independent transformation of multiple layers
 * - AWQActivationMultiply:     AWQ activation-scale Multiply support
 * - NegativeNoSoftmax:         Transformation skipped when Softmax is absent
 *
 * Qwen3 tests (create_qwen3_moe_graph):
 * - Qwen3BasicTransformation:         Gather insertion for Qwen3 SwiGLU expert
 * - RouterBroadcastChainShapeUpdate:  Reshape shape[0] updated after transpose replacement
 */

// Uncomment to save debug XML files during test execution
// #define SAVE_TEST_MODELS

namespace {

using namespace ov;
using namespace ov::npuw::pass;

// ============================================================================
// Test Utilities
// ============================================================================

class DeviceRoutedMoETransformTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Save model to XML for debugging
    void save_model(const std::shared_ptr<Model>& model, const std::string& prefix) {
#ifdef SAVE_TEST_MODELS
        std::string xml_path = prefix + ".xml";
        std::string bin_path = prefix + ".bin";
        ov::pass::Serialize serialize_pass(xml_path, bin_path);
        serialize_pass.run_on_model(const_cast<std::shared_ptr<Model>&>(model));
#endif
    }

    // Helper: Count nodes of specific type in model
    template <typename NodeType>
    size_t count_nodes(const std::shared_ptr<Model>& model) {
        size_t count = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (std::dynamic_pointer_cast<NodeType>(node)) {
                count++;
            }
        }
        return count;
    }

    // Helper: Find Gather nodes in model
    std::vector<std::shared_ptr<op::v8::Gather>> find_gather_nodes(const std::shared_ptr<Model>& model) {
        std::vector<std::shared_ptr<op::v8::Gather>> gathers;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto gather = std::dynamic_pointer_cast<op::v8::Gather>(node)) {
                gathers.push_back(gather);
            }
        }
        return gathers;
    }
};

// ============================================================================
// Synthetic Graph Builders
// ============================================================================

// Create complete Router graph based on router.ir
// Accepts shared input Parameter, returns router scores for Expert multiply
std::shared_ptr<op::v0::Unsqueeze> create_router_graph(const std::shared_ptr<op::v0::Parameter>& router_input,
                                                       int64_t k_value,
                                                       const std::string& layer_id,
                                                       size_t hidden_dim = 2880,
                                                       size_t num_experts = 32) {
    // Router MatMul with quantized weights (INT4 -> FP16 -> Multiply -> FP32)
    auto router_weights_int4 = op::v0::Constant::create(element::i4,
                                                        Shape{num_experts, hidden_dim},
                                                        std::vector<int8_t>(num_experts * hidden_dim, 1));
    router_weights_int4->set_friendly_name(layer_id + "mlp.router.weight_int4");

    auto router_weights_fp16 = std::make_shared<op::v0::Convert>(router_weights_int4, element::f16);
    auto router_scale =
        op::v0::Constant::create(element::f16, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto router_weights_scaled = std::make_shared<op::v1::Multiply>(router_weights_fp16, router_scale);
    auto router_weights_fp32 = std::make_shared<op::v0::Convert>(router_weights_scaled, element::f32);

    auto router_matmul = std::make_shared<op::v0::MatMul>(router_input, router_weights_fp32, false, true);
    router_matmul->set_friendly_name("__module.model." + layer_id + "mlp.router/aten::linear/MatMul");

    // Router Add (bias)
    auto router_bias =
        op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto router_add = std::make_shared<op::v1::Add>(router_matmul, router_bias);
    router_add->set_friendly_name("__module.model." + layer_id + "mlp.router/aten::linear/Add");

    // TopK: [1, num_experts] -> values [1, K], indices [1, K]
    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value});
    auto topk = std::make_shared<op::v11::TopK>(router_add,
                                                k_const,
                                                -1,
                                                op::v11::TopK::Mode::MAX,
                                                op::v11::TopK::SortType::NONE);
    topk->set_friendly_name("__module.model." + layer_id + "mlp.router/aten::topk/TopK");

    // Softmax on TopK values
    auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), 1);
    softmax->set_friendly_name("__module.model." + layer_id + "mlp.router/aten::softmax/Softmax");

    // ScatterElementsUpdate: scatter softmax back to [1, num_experts]
    auto zeros = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto indices_i32 = std::make_shared<op::v0::Convert>(topk->output(1), element::i32);
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{1});
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(zeros, indices_i32, softmax, scatter_axis);
    scatter->set_friendly_name("__module.model." + layer_id + "mlp.router/aten::scatter_/ScatterElementsUpdate");

    // Transpose: [1, num_experts] -> [num_experts, 1]
    auto transpose_order = op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, transpose_order);
    transpose->set_friendly_name("__module.model." + layer_id + "mlp.experts/aten::transpose/Transpose");

    // Reshape: [num_experts, 1] -> [num_experts, 1, 1]
    auto reshape_shape =
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1, 1});
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, reshape_shape, false);
    reshape->set_friendly_name("__module.model." + layer_id + "mlp.experts/aten::view/Reshape_2");

    // Unsqueeze: [num_experts, 1, 1] -> [num_experts, 1, 1, 1]
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{3});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(reshape, unsqueeze_axis);
    unsqueeze->set_friendly_name("__module.model." + layer_id + "mlp.experts/aten::unsqueeze/Unsqueeze_2");

    return unsqueeze;
}

// Create complete MoE graph with Router + Expert (GPT-OSS pattern)
// Router and Expert share the same input Parameter
std::shared_ptr<Model> create_complete_moe_graph(size_t num_experts = 32,
                                                 int64_t k_value = 4,
                                                 size_t hidden_dim = 2880,
                                                 size_t token_count = 1,
                                                 const std::string& layer_id = "layers.0.",
                                                 bool with_awq_multiply = false) {
    // 1. Create shared input Parameter for both Router and Expert
    auto shared_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{token_count, hidden_dim});
    shared_input->set_friendly_name(layer_id + "input");

    // 2. Create Router graph
    auto router_scores_output = create_router_graph(shared_input, k_value, layer_id, hidden_dim, num_experts);

    // 3. Create Expert graph (GPT-OSS Expert pattern)
    // Tile: [token_count, hidden_dim] -> [num_experts*token_count, hidden_dim]
    auto repeats =
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1});
    auto tile = std::make_shared<op::v0::Tile>(shared_input, repeats);
    tile->set_friendly_name("__module.model." + layer_id + "mlp.experts/Tile");

    // Reshape to 3D: [num_experts, token_count, hidden_dim]
    auto reshape_shape1 = op::v0::Constant::create(element::i64,
                                                   Shape{3},
                                                   std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                        static_cast<int64_t>(token_count),
                                                                        static_cast<int64_t>(hidden_dim)});
    auto reshape1 = std::make_shared<op::v1::Reshape>(tile, reshape_shape1, false);
    reshape1->set_friendly_name("__module.model." + layer_id + "mlp.experts/Reshape");

    // First MatMul (gate + up) with quantized weights [num_experts, hidden_dim*2, hidden_dim]
    auto weights_int4_1 = op::v0::Constant::create(element::i4,
                                                   Shape{num_experts, hidden_dim * 2, hidden_dim},
                                                   std::vector<int8_t>(num_experts * hidden_dim * 2 * hidden_dim, 1));
    weights_int4_1->set_friendly_name(layer_id + "mlp.experts.gate_up.weight_int4");

    auto weights_fp16_1 = std::make_shared<op::v0::Convert>(weights_int4_1, element::f16);
    auto weights_scale_1 = op::v0::Constant::create(element::f16,
                                                    Shape{num_experts, hidden_dim * 2, 1},
                                                    std::vector<float>(num_experts * hidden_dim * 2, 1.0f));
    auto weights_scaled_1 = std::make_shared<op::v1::Multiply>(weights_fp16_1, weights_scale_1);
    auto weights_fp32_1 = std::make_shared<op::v0::Convert>(weights_scaled_1, element::f32);

    auto matmul1 = std::make_shared<op::v0::MatMul>(reshape1, weights_fp32_1, false, true);
    matmul1->set_friendly_name("__module.model." + layer_id + "mlp.experts/MatMul_gate_up");

    auto biases1 = op::v0::Constant::create(element::f32,
                                            Shape{num_experts, 1, hidden_dim * 2},
                                            std::vector<float>(num_experts * hidden_dim * 2, 0.0f));
    auto add1 = std::make_shared<op::v1::Add>(matmul1, biases1);
    add1->set_friendly_name("__module.model." + layer_id + "mlp.experts/Add_gate_up");

    // Dual branches: Activation branch (Slice -> Minimum -> Swish) and Gate branch (Slice -> Clamp -> Add)
    // Activation branch
    auto slice_start1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto slice_stop1 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim)});
    auto slice_step1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto slice_axis1 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto slice1 = std::make_shared<op::v8::Slice>(add1, slice_start1, slice_stop1, slice_step1, slice_axis1);
    slice1->set_friendly_name("__module.model." + layer_id + "mlp.experts/Slice_activation");

    auto minimum_const = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{20.0f});
    auto minimum = std::make_shared<op::v1::Minimum>(slice1, minimum_const);
    minimum->set_friendly_name("__module.model." + layer_id + "mlp.experts/Minimum");

    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.0f});
    auto swish = std::make_shared<op::v4::Swish>(minimum, swish_beta);
    swish->set_friendly_name("__module.model." + layer_id + "mlp.experts/Swish");

    // Optional AWQ activation multiply (after Swish)
    std::shared_ptr<ov::Node> activation_output = swish;
    if (with_awq_multiply) {
        auto awq_scale = op::v0::Constant::create(element::f32,
                                                  Shape{num_experts, 1, hidden_dim},
                                                  std::vector<float>(num_experts * hidden_dim, 1.0f));
        awq_scale->set_friendly_name(layer_id + "mlp.experts.awq_scale");

        auto awq_multiply = std::make_shared<op::v1::Multiply>(swish, awq_scale);
        awq_multiply->set_friendly_name("__module.model." + layer_id + "mlp.experts/AWQMultiply");
        activation_output = awq_multiply;
    }

    // Gate branch
    auto slice_start2 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim)});
    auto slice_stop2 =
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{static_cast<int64_t>(hidden_dim * 2)});
    auto slice_step2 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto slice_axis2 = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{2});
    auto slice2 = std::make_shared<op::v8::Slice>(add1, slice_start2, slice_stop2, slice_step2, slice_axis2);
    slice2->set_friendly_name("__module.model." + layer_id + "mlp.experts/Slice_gate");

    auto clamp = std::make_shared<op::v0::Clamp>(slice2, -20.0f, 20.0f);
    clamp->set_friendly_name("__module.model." + layer_id + "mlp.experts/Clamp");

    auto add2_const = op::v0::Constant::create(element::f32, Shape{1}, std::vector<float>{0.0f});
    auto add2 = std::make_shared<op::v1::Add>(clamp, add2_const);
    add2->set_friendly_name("__module.model." + layer_id + "mlp.experts/Add_gate");

    // Merge branches: Multiply (use activation_output which may include AWQ multiply)
    auto multiply1 = std::make_shared<op::v1::Multiply>(activation_output, add2);
    multiply1->set_friendly_name("__module.model." + layer_id + "mlp.experts/Multiply_merge");

    // Second MatMul (down projection) with quantized weights [num_experts, hidden_dim, hidden_dim]
    auto weights_int4_2 = op::v0::Constant::create(element::i4,
                                                   Shape{num_experts, hidden_dim, hidden_dim},
                                                   std::vector<int8_t>(num_experts * hidden_dim * hidden_dim, 1));
    weights_int4_2->set_friendly_name(layer_id + "mlp.experts.down.weight_int4");

    auto weights_fp16_2 = std::make_shared<op::v0::Convert>(weights_int4_2, element::f16);
    auto weights_scale_2 = op::v0::Constant::create(element::f16,
                                                    Shape{num_experts, hidden_dim, 1},
                                                    std::vector<float>(num_experts * hidden_dim, 1.0f));
    auto weights_scaled_2 = std::make_shared<op::v1::Multiply>(weights_fp16_2, weights_scale_2);
    auto weights_fp32_2 = std::make_shared<op::v0::Convert>(weights_scaled_2, element::f32);

    auto matmul2 = std::make_shared<op::v0::MatMul>(multiply1, weights_fp32_2, false, true);
    matmul2->set_friendly_name("__module.model." + layer_id + "mlp.experts/MatMul_down");

    auto biases2 = op::v0::Constant::create(element::f32,
                                            Shape{num_experts, 1, hidden_dim},
                                            std::vector<float>(num_experts * hidden_dim, 0.0f));
    auto add3 = std::make_shared<op::v1::Add>(matmul2, biases2);
    add3->set_friendly_name("__module.model." + layer_id + "mlp.experts/Add_down");

    // Output reshape
    auto reshape_shape2 = op::v0::Constant::create(element::i64,
                                                   Shape{3},
                                                   std::vector<int64_t>{static_cast<int64_t>(num_experts),
                                                                        static_cast<int64_t>(token_count),
                                                                        static_cast<int64_t>(hidden_dim)});
    auto reshape2 = std::make_shared<op::v1::Reshape>(add3, reshape_shape2, false);
    reshape2->set_friendly_name("__module.model." + layer_id + "mlp.experts/Reshape_out");

    // Multiply with router scores
    auto output_multiply = std::make_shared<op::v1::Multiply>(reshape2, router_scores_output);
    output_multiply->set_friendly_name("__module.model." + layer_id + "mlp.experts/aten::mul/Multiply");

    // ReduceSum over expert dim — required by detect_router_by_topology step 3.
    auto reduce_axis = op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(output_multiply, reduce_axis);
    reduce_sum->set_friendly_name("__module.model." + layer_id + "mlp.experts/aten::sum/ReduceSum");

    auto result = std::make_shared<op::v0::Result>(reduce_sum);
    result->set_friendly_name(layer_id + "output");

    return std::make_shared<Model>(ResultVector{result}, ParameterVector{shared_input});
}

// ============================================================================
// Unit Tests
// ============================================================================

// Test 1: Basic transformation - Verify Gather insertion in quantized weights and shape updates
TEST_F(DeviceRoutedMoETransformTest, BasicTransformation) {
    constexpr size_t num_experts = 8;
    constexpr int64_t k_value = 4;
    constexpr size_t hidden_dim = 2880;

    auto model = create_complete_moe_graph(num_experts, k_value, hidden_dim, 1, "layers.0.");
    save_model(model, "device_routed_moe_basic_before");

    // Verify initial state
    EXPECT_EQ(count_nodes<op::v8::Gather>(model), 0) << "Should have no Gather before transformation";
    EXPECT_EQ(count_nodes<op::v0::Tile>(model), 1) << "Should have 1 Tile node";

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    manager.run_passes(model);

    // Validate
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    save_model(model, "device_routed_moe_basic_after");

    // Verify Gather insertion: gate_up weights + gate_up scale + gate_up biases + down weights + down scale + down
    // biases = 6
    auto gathers = find_gather_nodes(model);
    EXPECT_EQ(gathers.size(), 6) << "Should have 6 Gather nodes after transformation";

    // Verify Gather inserted in quantization chains (INT4->FP16->Gather->Multiply->FP32)
    // Note: Gather output may go through Convert before reaching Multiply
    size_t gathers_in_quant_chain = 0;
    for (const auto& gather : gathers) {
        // Check if Gather output feeds into Multiply (directly or through Convert)
        auto target_inputs = gather->output(0).get_target_inputs();
        if (target_inputs.empty())
            continue;

        auto next_node = target_inputs.begin()->get_node()->shared_from_this();

        // Direct path: Gather -> Multiply
        if (std::dynamic_pointer_cast<op::v1::Multiply>(next_node)) {
            gathers_in_quant_chain++;
            continue;
        }

        // Path with Convert: Gather -> Convert -> Multiply
        if (auto convert = std::dynamic_pointer_cast<op::v0::Convert>(next_node)) {
            auto convert_targets = convert->output(0).get_target_inputs();
            if (!convert_targets.empty()) {
                if (std::dynamic_pointer_cast<op::v1::Multiply>(
                        convert_targets.begin()->get_node()->shared_from_this())) {
                    gathers_in_quant_chain++;
                }
            }
        }
    }
    EXPECT_GE(gathers_in_quant_chain, 4)
        << "Gather should be inserted in quantization chains for gate_up and down weights (weights + scales)";

    // Verify Tile repeats updated from num_experts to k_value
    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<op::v0::Tile>(node)) {
            auto repeats_const =
                std::dynamic_pointer_cast<op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
            ASSERT_NE(repeats_const, nullptr);
            auto repeats_data = repeats_const->cast_vector<int64_t>();
            EXPECT_EQ(repeats_data[0], k_value) << "Tile repeats should be updated to K=" << k_value;
        }
    }

    // Verify Reshape shapes updated
    for (const auto& node : model->get_ordered_ops()) {
        if (auto reshape = std::dynamic_pointer_cast<op::v1::Reshape>(node)) {
            auto shape_const =
                std::dynamic_pointer_cast<op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
            if (shape_const) {
                auto shape_data = shape_const->cast_vector<int64_t>();
                if (shape_data.size() == 3 && shape_data[0] > 1) {
                    EXPECT_EQ(shape_data[0], k_value) << "Reshape expert dimension should be updated to K=" << k_value;
                }
            }
        }
    }
}

// Test 2: Multi-layer MoE
TEST_F(DeviceRoutedMoETransformTest, MultiLayerMoE) {
    constexpr size_t num_experts = 8;
    constexpr int64_t k_value_layer0 = 4;
    constexpr int64_t k_value_layer1 = 2;
    constexpr size_t hidden_dim = 2880;

    auto model_layer0 = create_complete_moe_graph(num_experts, k_value_layer0, hidden_dim, 1, "layers.0.");
    auto model_layer1 = create_complete_moe_graph(num_experts, k_value_layer1, hidden_dim, 1, "layers.1.");

    // Merge models
    ov::ParameterVector all_params;
    ov::ResultVector all_results;
    for (const auto& param : model_layer0->get_parameters())
        all_params.push_back(param);
    for (const auto& param : model_layer1->get_parameters())
        all_params.push_back(param);
    for (const auto& result : model_layer0->get_results())
        all_results.push_back(result);
    for (const auto& result : model_layer1->get_results())
        all_results.push_back(result);

    auto merged_model = std::make_shared<Model>(all_results, all_params);
    save_model(merged_model, "device_routed_moe_multi_layer_before");

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    manager.run_passes(merged_model);

    EXPECT_NO_THROW(merged_model->validate_nodes_and_infer_types());

    save_model(merged_model, "device_routed_moe_multi_layer_after");

    // Verify both layers transformed independently
    // Each layer: 6 Gathers (gate_up weights + scale + biases + down weights + scale + biases) = 12 total
    auto gathers = find_gather_nodes(merged_model);
    EXPECT_EQ(gathers.size(), 12) << "Should have 12 Gather nodes (6 per layer)";

    // Verify Tile repeats per layer
    size_t layer0_tiles = 0, layer1_tiles = 0;
    for (const auto& node : merged_model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<op::v0::Tile>(node)) {
            std::string name = tile->get_friendly_name();
            auto repeats_const =
                std::dynamic_pointer_cast<op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
            if (repeats_const) {
                auto repeats_data = repeats_const->cast_vector<int64_t>();
                if (name.find("layers.0.") != std::string::npos) {
                    EXPECT_EQ(repeats_data[0], k_value_layer0);
                    layer0_tiles++;
                } else if (name.find("layers.1.") != std::string::npos) {
                    EXPECT_EQ(repeats_data[0], k_value_layer1);
                    layer1_tiles++;
                }
            }
        }
    }
    EXPECT_EQ(layer0_tiles, 1);
    EXPECT_EQ(layer1_tiles, 1);
}

// Test 3: AWQ activation multiply support
TEST_F(DeviceRoutedMoETransformTest, AWQActivationMultiply) {
    constexpr size_t num_experts = 8;
    constexpr int64_t k_value = 4;
    constexpr size_t hidden_dim = 2880;

    auto model = create_complete_moe_graph(num_experts, k_value, hidden_dim, 1, "layers.0.", true);
    save_model(model, "device_routed_moe_awq_before");

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    manager.run_passes(model);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    save_model(model, "device_routed_moe_awq_after");

    // Verify AWQ multiply node exists
    bool found_awq_multiply = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto mult = std::dynamic_pointer_cast<op::v1::Multiply>(node)) {
            if (mult->get_friendly_name().find("AWQMultiply") != std::string::npos) {
                found_awq_multiply = true;

                // Verify one input is from Swish
                bool has_swish_input = false;
                for (size_t i = 0; i < 2; ++i) {
                    auto input = mult->input_value(i).get_node_shared_ptr();
                    if (std::dynamic_pointer_cast<op::v4::Swish>(input)) {
                        has_swish_input = true;
                        break;
                    }
                }
                EXPECT_TRUE(has_swish_input) << "AWQ Multiply should have Swish as input";

                // Verify other input is Gather (AWQ scale after transformation)
                bool has_gather_input = false;
                for (size_t i = 0; i < 2; ++i) {
                    auto input = mult->input_value(i).get_node_shared_ptr();
                    if (auto gather = std::dynamic_pointer_cast<op::v8::Gather>(input)) {
                        auto gather_shape = gather->get_output_shape(0);
                        EXPECT_EQ(gather_shape.size(), 3);
                        EXPECT_EQ(gather_shape[0], k_value) << "AWQ scale expert dimension should be K after Gather";
                        has_gather_input = true;
                        break;
                    }
                }
                EXPECT_TRUE(has_gather_input)
                    << "AWQ Multiply should have Gather (AWQ scale) as input after transformation";
                break;
            }
        }
    }
    EXPECT_TRUE(found_awq_multiply) << "AWQ Multiply node should be present when enabled";

    // Verify Gather inserted for AWQ scale
    auto gathers = find_gather_nodes(model);
    // gate_up weights + scale + biases + down weights + scale + biases + AWQ scale = 7
    EXPECT_EQ(gathers.size(), 7) << "Should have 7 Gather nodes (including AWQ scale)";
}

// Test 4: Negative test - No Softmax
TEST_F(DeviceRoutedMoETransformTest, NegativeNoSoftmax) {
    constexpr size_t num_experts = 8;
    constexpr int64_t k_value = 4;
    constexpr size_t hidden_dim = 2880;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value});
    auto topk =
        std::make_shared<op::v11::TopK>(input, k_const, 1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name("__module.model.layers.0.mlp.router/aten::topk/TopK");

    // NO Softmax - transformation should skip

    auto repeats =
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1});
    auto tile = std::make_shared<op::v0::Tile>(input, repeats);
    tile->set_friendly_name("__module.model.layers.0.mlp.experts/Tile");

    auto result = std::make_shared<op::v0::Result>(tile);
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
    save_model(model, "device_routed_moe_negative_before");

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    bool changed = manager.run_passes(model);

    // Should not transform
    EXPECT_FALSE(changed) << "Transformation should skip when Softmax is missing";

    auto repeats_const = std::dynamic_pointer_cast<op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
    auto repeats_data = repeats_const->cast_vector<int64_t>();
    EXPECT_EQ(repeats_data[0], num_experts) << "Tile repeats should remain unchanged";
}

// ============================================================================
// Qwen3 graph builder
// ============================================================================

// Qwen3-style MoE graph (decoding stage, token_count=1).
// Router: Input -> MatMul -> Softmax -> TopK -> ReduceSum -> Divide
//         -> Scatter -> Transpose -> Reshape([N,1,1]) -> Unsqueeze -> [N,1,1,1]
// Expert: Input -> Tile -> Reshape([N,1,H]) -> gate/up/down MatMuls (SwiGLU)
//         -> Reshape([N,1,1,H]) -> Multiply(router) -> ReduceSum
// Weights: Const(i4) -> Convert(f16) -> Multiply(scale) -> Convert(f32) -> MatMul
// 2 Gathers per MatMul (weight + scale) x 3 MatMuls = 6 total.
std::shared_ptr<Model> create_qwen3_moe_graph(size_t num_experts = 4,
                                              int64_t k_value = 2,
                                              size_t hidden_dim = 8,
                                              size_t intermediate_dim = 4) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    input->set_friendly_name("qwen3_input");

    // --- Router ---
    // NF4 weight chain: i4 [N,H] -> f16 -> Multiply(scale[N,1]) -> f32 -> MatMul
    auto rw = op::v0::Constant::create(element::i4,
                                       Shape{num_experts, hidden_dim},
                                       std::vector<int8_t>(num_experts * hidden_dim, 1));
    auto rw_f16 = std::make_shared<op::v0::Convert>(rw, element::f16);
    auto rs = op::v0::Constant::create(element::f16, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto rw_scaled = std::make_shared<op::v1::Multiply>(rw_f16, rs);
    auto rw_f32 = std::make_shared<op::v0::Convert>(rw_scaled, element::f32);
    auto router_mm = std::make_shared<op::v0::MatMul>(input, rw_f32, false, true);

    // Softmax -> TopK (Softmax BEFORE TopK, Qwen3 style)
    auto softmax = std::make_shared<op::v8::Softmax>(router_mm, 1);
    auto k_c = op::v0::Constant::create(element::i64, Shape{}, {k_value});
    auto topk =
        std::make_shared<op::v11::TopK>(softmax, k_c, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);

    // Renormalize: ReduceSum(values) -> Divide
    auto rs_ax = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduce_router = std::make_shared<op::v1::ReduceSum>(topk->output(0), rs_ax, true);
    auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce_router);

    // Scatter back to full expert dimension
    auto zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    auto bcast_shp = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1LL, (int64_t)num_experts});
    auto broadcast = std::make_shared<op::v3::Broadcast>(zero, bcast_shp);
    auto sc_ax = op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(broadcast, topk->output(1), divide, sc_ax);

    // Transpose [1,N] -> [N,1]
    auto tp_ord = op::v0::Constant::create(element::i32, Shape{2}, {1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, tp_ord);

    // Reshape [N,1] -> [N,1,1]  (shape[0] = num_experts, fixed by transform_router_broadcast_chain)
    auto r_shp = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{(int64_t)num_experts, 1LL, 1LL});
    auto router_reshape = std::make_shared<op::v1::Reshape>(transpose, r_shp, false);

    // Unsqueeze dim 3: [N,1,1] -> [N,1,1,1]
    auto us_ax = op::v0::Constant::create(element::i64, Shape{1}, {3});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(router_reshape, us_ax);

    // --- Expert ---
    // Helper: build INT4 weight dequantization chain: Const(i4,[d0,d1,d2])->f16->Multiply(scale)->f32
    auto make_weight = [](size_t d0, size_t d1, size_t d2) {
        auto w = op::v0::Constant::create(element::i4, Shape{d0, d1, d2}, std::vector<int8_t>(d0 * d1 * d2, 1));
        auto wf = std::make_shared<op::v0::Convert>(w, element::f16);
        auto sc = op::v0::Constant::create(element::f16, Shape{d0, d1, 1}, std::vector<float>(d0 * d1, 1.0f));
        auto ws = std::make_shared<op::v1::Multiply>(wf, sc);
        return std::make_shared<op::v0::Convert>(ws, element::f32);
    };

    // Tile [1,H] -> [N,H]
    auto tile_rep = op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{(int64_t)num_experts, 1LL});
    auto tile = std::make_shared<op::v0::Tile>(input, tile_rep);

    // Reshape [N,H] -> [N,1,H]  (shape[0] = num_experts)
    auto rs1_shp = op::v0::Constant::create(element::i64,
                                            Shape{3},
                                            std::vector<int64_t>{(int64_t)num_experts, 1LL, (int64_t)hidden_dim});
    auto reshape1 = std::make_shared<op::v1::Reshape>(tile, rs1_shp, false);

    // Gate: [N,I,H] weights -> [N,1,H] x [N,I,H]^T -> [N,1,I] -> Swish
    auto gate_w = make_weight(num_experts, intermediate_dim, hidden_dim);
    auto matmul_gate = std::make_shared<op::v0::MatMul>(reshape1, gate_w, false, true);
    auto swish = std::make_shared<op::v4::Swish>(matmul_gate);

    // Up: [N,I,H] weights -> [N,1,H] x [N,I,H]^T -> [N,1,I]
    auto up_w = make_weight(num_experts, intermediate_dim, hidden_dim);
    auto matmul_up = std::make_shared<op::v0::MatMul>(reshape1, up_w, false, true);

    // SwiGLU
    auto swiglu = std::make_shared<op::v1::Multiply>(swish, matmul_up);

    // Down: [N,H,I] weights -> [N,1,I] x [N,H,I]^T -> [N,1,H]
    auto down_w = make_weight(num_experts, hidden_dim, intermediate_dim);
    auto matmul_down = std::make_shared<op::v0::MatMul>(swiglu, down_w, false, true);

    // Reshape [N,1,H] -> [N,1,1,H]  (shape[0] = num_experts)
    auto rs2_shp = op::v0::Constant::create(element::i64,
                                            Shape{4},
                                            std::vector<int64_t>{(int64_t)num_experts, 1LL, 1LL, (int64_t)hidden_dim});
    auto reshape2 = std::make_shared<op::v1::Reshape>(matmul_down, rs2_shp, false);

    // Output multiply: [N,1,1,H] * [N,1,1,1] -> [N,1,1,H]
    auto output_multiply = std::make_shared<op::v1::Multiply>(reshape2, unsqueeze);

    // ReduceSum over expert dim — required by detect_router_by_topology step 3.
    auto final_ax = op::v0::Constant::create(element::i64, Shape{1}, {0});
    auto final_reduce = std::make_shared<op::v1::ReduceSum>(output_multiply, final_ax);

    auto result = std::make_shared<op::v0::Result>(final_reduce);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input});
}

// ============================================================================
// Qwen3 tests
// ============================================================================

// Test 5: Qwen3 basic transformation
// Verifies 6 Gather nodes inserted, Tile repeats[0] and Reshape dim[0] updated to k_value.
TEST_F(DeviceRoutedMoETransformTest, Qwen3BasicTransformation) {
    constexpr size_t num_experts = 4;
    constexpr int64_t k_value = 2;
    constexpr size_t hidden_dim = 8;
    constexpr size_t intermediate_dim = 4;

    auto model = create_qwen3_moe_graph(num_experts, k_value, hidden_dim, intermediate_dim);
    save_model(model, "qwen3_moe_basic_before");

    EXPECT_EQ(count_nodes<op::v8::Gather>(model), 0u) << "No Gather before transformation";
    EXPECT_EQ(count_nodes<op::v0::Tile>(model), 1u) << "One Tile before transformation";

    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    manager.run_passes(model);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    save_model(model, "qwen3_moe_basic_after");

    // 2 Gathers per MatMul (weight + scale) x 3 MatMuls = 6
    auto gathers = find_gather_nodes(model);
    EXPECT_EQ(gathers.size(), 6u) << "Expected 6 Gather nodes (weight+scale per gate/up/down MatMul)";

    // Tile repeats[0] must be updated to k_value
    for (const auto& node : model->get_ordered_ops()) {
        if (auto tile = std::dynamic_pointer_cast<op::v0::Tile>(node)) {
            auto rep = std::dynamic_pointer_cast<op::v0::Constant>(tile->input_value(1).get_node_shared_ptr());
            ASSERT_NE(rep, nullptr);
            EXPECT_EQ(rep->cast_vector<int64_t>()[0], k_value) << "Tile repeats[0] should be updated to k=" << k_value;
        }
    }

    // All Reshape shape constants where dim[0] > 1 must be updated to k_value
    for (const auto& node : model->get_ordered_ops()) {
        if (auto reshape = std::dynamic_pointer_cast<op::v1::Reshape>(node)) {
            auto shp = std::dynamic_pointer_cast<op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
            if (!shp)
                continue;
            auto d = shp->cast_vector<int64_t>();
            if (!d.empty() && d[0] > 1) {
                EXPECT_EQ(d[0], k_value) << "Reshape expert dim should be updated to k=" << k_value;
            }
        }
    }
}

// Test 6: Router broadcast chain Reshape shape fix (Qwen3-specific)
// After transform_transpose replaces Scatter->Transpose with router_scores [1,k],
// transform_router_broadcast_chain must update Reshape shape[0] from num_experts to k_value.
TEST_F(DeviceRoutedMoETransformTest, RouterBroadcastChainShapeUpdate) {
    constexpr size_t num_experts = 4;
    constexpr int64_t k_value = 2;
    constexpr size_t hidden_dim = 8;
    constexpr size_t intermediate_dim = 4;

    auto model = create_qwen3_moe_graph(num_experts, k_value, hidden_dim, intermediate_dim);

    // Before: locate the 3-element Reshape [num_experts, 1, 1] in the router broadcast chain
    auto find_router_reshape_shape = [&]() -> std::shared_ptr<op::v0::Constant> {
        for (const auto& node : model->get_ordered_ops()) {
            if (auto reshape = std::dynamic_pointer_cast<op::v1::Reshape>(node)) {
                auto shp = std::dynamic_pointer_cast<op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
                if (!shp)
                    continue;
                auto d = shp->cast_vector<int64_t>();
                if (d.size() == 3 && d[0] == (int64_t)num_experts && d[1] == 1 && d[2] == 1)
                    return shp;
            }
        }
        return nullptr;
    };

    auto before_shp = find_router_reshape_shape();
    ASSERT_NE(before_shp, nullptr) << "Router broadcast Reshape [N,1,1] should exist before transform";
    EXPECT_EQ(before_shp->cast_vector<int64_t>()[0], (int64_t)num_experts);

    ov::pass::Manager manager;
    manager.register_pass<DeviceRoutedMoETransform>();
    manager.run_passes(model);

    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    // After: the Reshape in the chain must now have shape [k_value, 1, 1]
    bool found = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto reshape = std::dynamic_pointer_cast<op::v1::Reshape>(node)) {
            auto shp = std::dynamic_pointer_cast<op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
            if (!shp)
                continue;
            auto d = shp->cast_vector<int64_t>();
            if (d.size() == 3 && d[0] == k_value && d[1] == 1 && d[2] == 1) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found) << "Router broadcast Reshape shape[0] must be updated from " << num_experts
                       << " to k=" << k_value;
}

}  // namespace
