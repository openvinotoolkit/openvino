// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_transformations/gather_to_2d_gather.hpp"

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

/*
 * Test suite for GatherTo2DGather Transformation
 *
 * Testing Strategy:
 * - BasicTransformation: Verify 3D->2D Gather transformation and output shapes
 * - NegativeAxis: Ensure transformation skips when axis != 0
 * - Negative2DData: Ensure transformation skips for 1D/2D data
 * - NegativeSingleDimension: Ensure transformation skips when M=1 or K=1
 * - LargeDimensions: Stress test with realistic MoE sizes
 */

// Uncomment to save debug XML files during test execution
// #define SAVE_TEST_MODELS

namespace {

using namespace ov;
using namespace ov::npuw::pass;

// ============================================================================
// Test Utilities
// ============================================================================

class GatherTo2DGatherTest : public ::testing::Test {
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

    // Helper: Validate transformation results
    void validate_transformation(const std::shared_ptr<Model>& model, int64_t I, int64_t M, int64_t K) {
        // Verify node counts
        EXPECT_EQ(count_nodes<op::v8::Gather>(model), 1) << "Should have 1 Gather after transformation";
        EXPECT_EQ(count_nodes<op::v0::Tile>(model), 1) << "Should have 1 Tile";
        EXPECT_EQ(count_nodes<op::v1::Multiply>(model), 1) << "Should have 1 Multiply";
        EXPECT_EQ(count_nodes<op::v1::Add>(model), 1) << "Should have 1 Add";

        // Verify Multiply constant is M
        bool found_multiply_constant = false;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto multiply = std::dynamic_pointer_cast<op::v1::Multiply>(node)) {
                for (size_t i = 0; i < 2; ++i) {
                    auto input = multiply->input_value(i).get_node_shared_ptr();
                    if (auto constant = std::dynamic_pointer_cast<op::v0::Constant>(input)) {
                        auto constant_data = constant->cast_vector<int64_t>();
                        if (constant_data.size() == 1 && constant_data[0] == M) {
                            found_multiply_constant = true;
                            break;
                        }
                    }
                }
            }
        }
        EXPECT_TRUE(found_multiply_constant) << "Multiply should have M=" << M << " as constant";

        // Verify Add constant is range [0, 1, 2, ..., M-1]
        bool found_range_constant = false;
        for (const auto& node : model->get_ordered_ops()) {
            auto add = std::dynamic_pointer_cast<op::v1::Add>(node);
            if (!add)
                continue;

            for (size_t i = 0; i < 2; ++i) {
                auto input = add->input_value(i).get_node_shared_ptr();
                auto tile = std::dynamic_pointer_cast<op::v0::Tile>(input);
                if (!tile)
                    continue;

                auto range_input = tile->input_value(0).get_node_shared_ptr();
                auto constant = std::dynamic_pointer_cast<op::v0::Constant>(range_input);
                if (!constant)
                    continue;

                auto range_data = constant->cast_vector<int64_t>();
                if (range_data.size() != static_cast<size_t>(M))
                    continue;

                bool valid_range = true;
                for (size_t j = 0; j < range_data.size(); ++j) {
                    if (range_data[j] != static_cast<int64_t>(j)) {
                        valid_range = false;
                        break;
                    }
                }

                if (valid_range) {
                    found_range_constant = true;
                    break;
                }
            }

            if (found_range_constant)
                break;
        }
        EXPECT_TRUE(found_range_constant) << "Add should have range [0, M-1] constant";

        // Verify output shape from model results
        auto results = model->get_results();
        ASSERT_EQ(results.size(), 1) << "Model should have 1 result";
        auto output_shape = results[0]->get_output_shape(0);

        ASSERT_EQ(output_shape.size(), 3) << "Output should be 3D";
        EXPECT_EQ(output_shape[0], I) << "First dimension should be I (num_selected)";
        EXPECT_EQ(output_shape[1], M) << "Second dimension should be M (feature_dim)";
        EXPECT_EQ(output_shape[2], K) << "Third dimension should be K (hidden_dim)";
    }
};

// ============================================================================
// Synthetic Graph Builders
// ============================================================================

// Create a simple 3D Gather graph
// data: [N, M, K], indices: [I], axis: 0 -> output: [I, M, K]
std::shared_ptr<Model> create_3d_gather_graph(int64_t N,
                                              int64_t M,
                                              int64_t K,
                                              int64_t I,
                                              const std::string& name_prefix = "gather") {
    // Data input [N, M, K]
    auto data = op::v0::Constant::create(element::f32,
                                         Shape{static_cast<size_t>(N), static_cast<size_t>(M), static_cast<size_t>(K)},
                                         std::vector<float>(N * M * K, 1.0f));
    data->set_friendly_name(name_prefix + "_data");

    // Indices [I]
    std::vector<int64_t> indices_data(I);
    for (int64_t i = 0; i < I; ++i) {
        indices_data[i] = i % N;  // Valid indices within [0, N)
    }
    auto indices = op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(I)}, indices_data);
    indices->set_friendly_name(name_prefix + "_indices");

    // Axis = 0
    auto axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});

    // Gather
    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    gather->set_friendly_name(name_prefix);

    // Result
    auto result = std::make_shared<op::v0::Result>(gather);
    result->set_friendly_name(name_prefix + "_output");

    return std::make_shared<Model>(ResultVector{result}, ParameterVector{});
}

// Create Gather graph with non-zero axis
std::shared_ptr<Model> create_gather_with_axis(int64_t axis_value) {
    auto data = op::v0::Constant::create(element::f32, Shape{8, 16, 32}, std::vector<float>(8 * 16 * 32, 1.0f));
    auto indices = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{axis_value});

    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    gather->set_friendly_name("gather_axis_" + std::to_string(axis_value));

    auto result = std::make_shared<op::v0::Result>(gather);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{});
}

// Create Gather graph with 2D data
std::shared_ptr<Model> create_gather_with_2d_data() {
    auto data = op::v0::Constant::create(element::f32, Shape{8, 32}, std::vector<float>(8 * 32, 1.0f));
    auto indices = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});

    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    gather->set_friendly_name("gather_2d");

    auto result = std::make_shared<op::v0::Result>(gather);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{});
}

// Create Gather graph with M=1 (single feature dimension)
std::shared_ptr<Model> create_gather_with_single_m() {
    auto data = op::v0::Constant::create(element::f32, Shape{8, 1, 32}, std::vector<float>(8 * 1 * 32, 1.0f));
    auto indices = op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{0});

    auto gather = std::make_shared<op::v8::Gather>(data, indices, axis);
    gather->set_friendly_name("gather_m1");

    auto result = std::make_shared<op::v0::Result>(gather);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{});
}

// ============================================================================
// Unit Tests
// ============================================================================

// Test 1: Basic transformation - Verify 3D->2D Gather transformation
TEST_F(GatherTo2DGatherTest, BasicTransformation) {
    constexpr int64_t N = 8;   // num_experts
    constexpr int64_t M = 16;  // feature_dim
    constexpr int64_t K = 32;  // hidden_dim
    constexpr int64_t I = 4;   // num_selected

    auto model = create_3d_gather_graph(N, M, K, I);
    save_model(model, "gather_to_2d_basic_before");

    // Verify initial state
    EXPECT_EQ(count_nodes<op::v8::Gather>(model), 1) << "Should have 1 Gather before transformation";
    EXPECT_EQ(count_nodes<op::v1::Reshape>(model), 0) << "Should have no Reshape before transformation";
    EXPECT_EQ(count_nodes<op::v0::Tile>(model), 0) << "Should have no Tile before transformation";

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<GatherTo2DGather>();
    bool changed = manager.run_passes(model);

    EXPECT_TRUE(changed) << "Transformation should modify the graph";
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    save_model(model, "gather_to_2d_basic_after");

    // Validate transformation results
    validate_transformation(model, I, M, K);
}

// Test 2: Negative test - Axis != 0
TEST_F(GatherTo2DGatherTest, NegativeAxis) {
    auto model = create_gather_with_axis(1);  // axis = 1
    save_model(model, "gather_to_2d_negative_axis_before");

    ov::pass::Manager manager;
    manager.register_pass<GatherTo2DGather>();
    bool changed = manager.run_passes(model);

    EXPECT_FALSE(changed) << "Transformation should skip when axis != 0";
}

// Test 3: Negative test - 2D data
TEST_F(GatherTo2DGatherTest, Negative2DData) {
    auto model = create_gather_with_2d_data();
    save_model(model, "gather_to_2d_negative_2d_before");

    ov::pass::Manager manager;
    manager.register_pass<GatherTo2DGather>();
    bool changed = manager.run_passes(model);

    EXPECT_FALSE(changed) << "Transformation should skip for 2D data";
}

// Test 4: Negative test - M=1 (transformation not beneficial)
TEST_F(GatherTo2DGatherTest, NegativeSingleDimension) {
    auto model = create_gather_with_single_m();
    save_model(model, "gather_to_2d_negative_m1_before");

    ov::pass::Manager manager;
    manager.register_pass<GatherTo2DGather>();
    bool changed = manager.run_passes(model);

    EXPECT_FALSE(changed) << "Transformation should skip when M=1 (not beneficial)";
}

// Test 5: Large dimensions - Stress test with realistic MoE sizes
TEST_F(GatherTo2DGatherTest, LargeDimensions) {
    constexpr int64_t N = 32;    // 32 experts
    constexpr int64_t M = 2880;  // feature_dim (typical for large models)
    constexpr int64_t K = 2880;  // hidden_dim
    constexpr int64_t I = 4;     // top-4 routing

    auto model = create_3d_gather_graph(N, M, K, I);
    save_model(model, "gather_to_2d_large_before");

    ov::pass::Manager manager;
    manager.register_pass<GatherTo2DGather>();
    bool changed = manager.run_passes(model);

    EXPECT_TRUE(changed) << "Transformation should handle large dimensions";
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    save_model(model, "gather_to_2d_large_after");

    // Validate transformation results
    validate_transformation(model, I, M, K);
}

}  // namespace
