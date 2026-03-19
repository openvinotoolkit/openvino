// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_kvcache_into_blocks.hpp"

#include <gtest/gtest.h>

#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace ov::npuw::pass;

class SplitKVCacheIntoBlocksTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test setup
    }

    // Helper: Create a simple model with KV cache parameter -> Concat
    std::shared_ptr<ov::Model> create_kv_cache_model(const std::string& param_name,
                                                     const ov::Shape& shape,
                                                     int64_t concat_axis) {
        // Create KV cache parameter
        auto kv_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape);
        kv_param->set_friendly_name(param_name);

        // Create new token parameter (to concatenate with)
        ov::Shape new_token_shape = shape;
        new_token_shape[concat_axis] = 1;  // Single token
        auto new_token = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, new_token_shape);
        new_token->set_friendly_name("new_token");

        // Create Concat
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv_param, new_token}, concat_axis);

        // Create Result
        auto result = std::make_shared<ov::op::v0::Result>(concat);

        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{kv_param, new_token});
    }
};

TEST_F(SplitKVCacheIntoBlocksTest, TransformKeyParameter) {
    // Test: Transform past_key parameter with axis=2
    // Original: past_key [1, 32, 64, 128] + new_token [1, 32, 1, 128]
    // After: 4 blocks of 16 tokens each + new_token
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 64, 128};        // [B, H, S=64, D]
    const uint32_t expected_blocks = 64 / block_size;  // 4 blocks

    auto model = create_kv_cache_model("past_key.0", orig_shape, 2);

    // Apply transformation (max_blocks removed, auto-calculated from shape)
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Should have expected_blocks + 1 parameters (blocks + new_token)
    EXPECT_EQ(model->get_parameters().size(), expected_blocks + 1);

    // Verify: Each block parameter has correct shape [1, 32, 16, 128]
    uint32_t block_count = 0;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_") != std::string::npos) {
            block_count++;
            auto block_shape = param->get_shape();
            EXPECT_EQ(block_shape[0], 1);           // batch
            EXPECT_EQ(block_shape[1], 32);          // num_heads
            EXPECT_EQ(block_shape[2], block_size);  // block_size
            EXPECT_EQ(block_shape[3], 128);         // head_dim
        }
    }
    EXPECT_EQ(block_count, expected_blocks);

    // Verify: Concat has blocks + new_token input (CRITICAL: present_key preserved!)
    bool found_concat = false;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            found_concat = true;
            auto concat = ov::as_type_ptr<ov::op::v0::Concat>(op);
            EXPECT_EQ(concat->get_axis(), 2);
            EXPECT_EQ(concat->get_input_size(), expected_blocks + 1);  // blocks + present_key
        }
    }
    EXPECT_TRUE(found_concat);
}

TEST_F(SplitKVCacheIntoBlocksTest, TransformValueParameterTransposed) {
    // Test: Transform past_value parameter with v_transposed=true (axis=3)
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 128, 64};  // [B, H, D, S] - transposed

    auto model = create_kv_cache_model("past_value.0", orig_shape, 3);

    // Apply transformation with v_transposed=true
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    const uint32_t expected_blocks = 64 / block_size;  // 4 blocks

    // Verify: Block parameters have correct shape [1, 32, 128, 16]
    uint32_t block_count = 0;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_") != std::string::npos) {
            block_count++;
            auto block_shape = param->get_shape();
            EXPECT_EQ(block_shape[0], 1);           // batch
            EXPECT_EQ(block_shape[1], 32);          // num_heads
            EXPECT_EQ(block_shape[2], 128);         // head_dim
            EXPECT_EQ(block_shape[3], block_size);  // block_size
        }
    }
    EXPECT_EQ(block_count, expected_blocks);

    // Verify: Concat axis=3 and has blocks + present_value
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            auto concat = ov::as_type_ptr<ov::op::v0::Concat>(op);
            EXPECT_EQ(concat->get_axis(), 3);
            EXPECT_EQ(concat->get_input_size(), expected_blocks + 1);
        }
    }
}

TEST_F(SplitKVCacheIntoBlocksTest, TransformValueParameterNotTransposed) {
    // Test: Transform past_value parameter with v_transposed=false (axis=2)
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 64, 128};  // [B, H, S, D] - same as K

    auto model = create_kv_cache_model("past_value.0", orig_shape, 2);

    // Apply transformation with v_transposed=false
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, false);
    manager.run_passes(model);

    const uint32_t expected_blocks = 64 / block_size;  // 4 blocks

    // Verify: Block parameters have correct shape [1, 32, 16, 128]
    uint32_t block_count = 0;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_") != std::string::npos) {
            block_count++;
            auto block_shape = param->get_shape();
            EXPECT_EQ(block_shape[0], 1);           // batch
            EXPECT_EQ(block_shape[1], 32);          // num_heads
            EXPECT_EQ(block_shape[2], block_size);  // block_size (at axis=2)
            EXPECT_EQ(block_shape[3], 128);         // head_dim
        }
    }
    EXPECT_EQ(block_count, expected_blocks);

    // Verify: Concat axis=2 (same as K) and has blocks + present_value
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            auto concat = ov::as_type_ptr<ov::op::v0::Concat>(op);
            EXPECT_EQ(concat->get_axis(), 2);
            EXPECT_EQ(concat->get_input_size(), expected_blocks + 1);
        }
    }
}

TEST_F(SplitKVCacheIntoBlocksTest, AlternativeNaming) {
    // Test: Transform past_key_values.0.key naming convention
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 64, 128};

    auto model = create_kv_cache_model("past_key_values.0.key", orig_shape, 2);

    const uint32_t expected_blocks = 64 / block_size;  // 4 blocks

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Should have transformed
    EXPECT_EQ(model->get_parameters().size(), expected_blocks + 1);

    bool found_blocks = false;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_") != std::string::npos) {
            found_blocks = true;
        }
    }
    EXPECT_TRUE(found_blocks);
}

TEST_F(SplitKVCacheIntoBlocksTest, SkipNonKVCacheParameter) {
    // Test: Should NOT transform non-KV cache parameters
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 64, 128};

    auto model = create_kv_cache_model("hidden_states", orig_shape, 2);

    size_t orig_param_count = model->get_parameters().size();

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Parameter count unchanged (not transformed)
    EXPECT_EQ(model->get_parameters().size(), orig_param_count);

    // Verify: No block parameters created
    for (const auto& param : model->get_parameters()) {
        EXPECT_EQ(param->get_friendly_name().find("_block_"), std::string::npos);
    }
}

TEST_F(SplitKVCacheIntoBlocksTest, InvalidRank) {
    // Test: Should skip parameters with invalid rank (not 4D)
    const uint32_t block_size = 16;
    const ov::Shape invalid_shape{1, 32, 64};  // 3D instead of 4D

    auto kv_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, invalid_shape);
    kv_param->set_friendly_name("past_key.0");

    ov::Shape new_token_shape{1, 32, 1};
    auto new_token = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, new_token_shape);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{kv_param, new_token}, 2);
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{kv_param, new_token});

    size_t orig_param_count = model->get_parameters().size();

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Not transformed due to invalid rank
    EXPECT_EQ(model->get_parameters().size(), orig_param_count);
}

TEST_F(SplitKVCacheIntoBlocksTest, MultipleKVParameters) {
    // Test: Transform multiple KV cache parameters in one model
    const uint32_t block_size = 16;
    const uint32_t expected_blocks = 64 / block_size;  // 4 blocks per parameter

    // Create model with both past_key and past_value
    auto key_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 64, 128});
    key_param->set_friendly_name("past_key.0");

    auto value_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 128, 64});
    value_param->set_friendly_name("past_value.0");

    auto new_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 1, 128});
    auto new_v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 32, 128, 1});

    auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{key_param, new_k}, 2);
    auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{value_param, new_v}, 3);

    auto result_k = std::make_shared<ov::op::v0::Result>(concat_k);
    auto result_v = std::make_shared<ov::op::v0::Result>(concat_v);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_k, result_v},
                                             ov::ParameterVector{key_param, value_param, new_k, new_v});

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Should have (expected_blocks * 2) + 2 parameters (K blocks + V blocks + new_k + new_v)
    EXPECT_EQ(model->get_parameters().size(), (expected_blocks * 2) + 2);

    // Count block parameters
    size_t key_blocks = 0, value_blocks = 0;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("past_key") != std::string::npos &&
            param->get_friendly_name().find("_block_") != std::string::npos) {
            key_blocks++;
        }
        if (param->get_friendly_name().find("past_value") != std::string::npos &&
            param->get_friendly_name().find("_block_") != std::string::npos) {
            value_blocks++;
        }
    }

    EXPECT_EQ(key_blocks, expected_blocks);
    EXPECT_EQ(value_blocks, expected_blocks);
}

TEST_F(SplitKVCacheIntoBlocksTest, TailBlockHandling) {
    // Test: Handle non-evenly divisible seq_len (creates tail block)
    // seq_len=70, block_size=16 -> 4 full blocks (64 tokens) + 1 tail block (6 tokens)
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 70, 128};                                                      // [B, H, S=70, D]
    const uint32_t expected_full_blocks = 70 / block_size;                                           // 4
    const uint32_t expected_tail_size = 70 % block_size;                                             // 6
    const uint32_t expected_total_blocks = expected_full_blocks + (expected_tail_size > 0 ? 1 : 0);  // 5

    auto model = create_kv_cache_model("past_key.0", orig_shape, 2);

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Should have 5 block parameters + 1 new_token = 6 total
    EXPECT_EQ(model->get_parameters().size(), expected_total_blocks + 1);

    // Verify: 4 full blocks + 1 tail block
    uint32_t full_block_count = 0;
    bool found_tail_block = false;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_tail") != std::string::npos) {
            found_tail_block = true;
            auto tail_shape = param->get_shape();
            EXPECT_EQ(tail_shape[2], expected_tail_size);  // tail block has 6 tokens
        } else if (param->get_friendly_name().find("_block_") != std::string::npos) {
            full_block_count++;
            auto block_shape = param->get_shape();
            EXPECT_EQ(block_shape[2], block_size);  // full block has 16 tokens
        }
    }
    EXPECT_EQ(full_block_count, expected_full_blocks);
    EXPECT_TRUE(found_tail_block);

    // Verify: Concat has all blocks + present_key
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            auto concat = ov::as_type_ptr<ov::op::v0::Concat>(op);
            EXPECT_EQ(concat->get_input_size(), expected_total_blocks + 1);  // 5 blocks + present_key
        }
    }
}

TEST_F(SplitKVCacheIntoBlocksTest, WithConvertNode) {
    // Test: Transform KV cache with Convert node between Parameter and Concat
    // Pattern: Parameter(f16) -> Convert(f32) -> Concat
    const uint32_t block_size = 16;
    const ov::Shape orig_shape{1, 32, 64, 128};  // [B, H, S=64, D]

    // Create KV cache parameter
    auto kv_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, orig_shape);
    kv_param->set_friendly_name("past_key.0");

    // Create Convert node (f16 -> f32)
    auto convert = std::make_shared<ov::op::v0::Convert>(kv_param, ov::element::f32);

    // Create new token parameter
    ov::Shape new_token_shape = orig_shape;
    new_token_shape[2] = 1;  // Single token
    auto new_token = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, new_token_shape);
    new_token->set_friendly_name("new_token");

    // Create Concat with Convert output and new_token
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{convert, new_token}, 2);

    // Create Result
    auto result = std::make_shared<ov::op::v0::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{kv_param, new_token});

    // Apply transformation
    ov::pass::Manager manager;
    manager.register_pass<SplitKVCacheIntoBlocks>(block_size, true);
    manager.run_passes(model);

    // Verify: Should have 4 block parameters + 1 new_token
    EXPECT_EQ(model->get_parameters().size(), 5);

    // Verify: Each block parameter should be followed by a Convert node
    uint32_t block_count = 0;
    uint32_t convert_count = 0;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name().find("_block_") != std::string::npos) {
            block_count++;
            // Check that parameter is f16
            EXPECT_EQ(param->get_element_type(), ov::element::f16);

            // Check that it's followed by a Convert node
            for (const auto& output : param->outputs()) {
                for (const auto& input : output.get_target_inputs()) {
                    auto node = input.get_node()->shared_from_this();
                    if (ov::is_type<ov::op::v0::Convert>(node)) {
                        auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(node);
                        EXPECT_EQ(cvt->get_destination_type(), ov::element::f32);
                        convert_count++;
                    }
                }
            }
        }
    }
    EXPECT_EQ(block_count, 4);
    EXPECT_EQ(convert_count, 4);  // Each block should have a Convert

    // Verify: Concat receives f32 inputs from Convert nodes
    bool found_concat = false;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Concat>(op)) {
            found_concat = true;
            auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(op);
            EXPECT_EQ(concat_node->get_input_size(), 5);  // 4 blocks + new_token

            // Verify all concat inputs are f32
            for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
                EXPECT_EQ(concat_node->get_input_element_type(i), ov::element::f32);
            }
        }
    }
    EXPECT_TRUE(found_concat);
}
