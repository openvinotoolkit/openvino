// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::pattern::op;

// Simple single-node block match
TEST(PatternBlockTest, block_matches_simple_matmul) {
    auto input = any_input();
    auto weights = any_input();
    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weights});

    auto block = std::make_shared<Block>(OutputVector{input, weights}, OutputVector{matmul}, "matmul_block");

    Matcher matcher(block);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto real_matmul = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);

    ASSERT_TRUE(matcher.match(real_matmul->output(0)));

    auto block_node = ov::as_type_ptr<Block>(matcher.get_pattern_value_map().at(block).get_node_shared_ptr());
    ASSERT_NE(block_node, nullptr);
    EXPECT_EQ(block_node->get_inputs()[0].get_node_shared_ptr(), input_node);
    EXPECT_EQ(block_node->get_inputs()[1].get_node_shared_ptr(), weights_node);
    EXPECT_EQ(block_node->get_outputs()[0].get_node_shared_ptr(), real_matmul);
}

// Block matching MatMul followed by Add
TEST(PatternBlockTest, block_matches_matmul_add_chain) {
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weights});
    auto add = wrap_type<ov::op::v1::Add>({matmul, bias});

    auto block = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "matmul_add_block");

    Matcher matcher(block);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto bias_node = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, {0});

    auto matmul_node = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);
    auto add_node = std::make_shared<ov::op::v1::Add>(matmul_node, bias_node);

    ASSERT_TRUE(matcher.match(add_node->output(0)));

    auto block_node = ov::as_type_ptr<Block>(matcher.get_pattern_value_map().at(block).get_node_shared_ptr());
    ASSERT_NE(block_node, nullptr);
    EXPECT_EQ(block_node->get_outputs()[0].get_node_shared_ptr(), add_node);
}

// Block fails to match if operator type differs (Subtract instead of Add)
TEST(PatternBlockTest, block_does_not_match_wrong_operator) {
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weights});
    auto add = wrap_type<ov::op::v1::Add>({matmul, bias});

    auto block = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "matmul_add_block");

    Matcher matcher(block);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto bias_node = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, {0});

    auto matmul_node = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);
    auto subtract_node = std::make_shared<ov::op::v1::Subtract>(matmul_node, bias_node);

    ASSERT_FALSE(matcher.match(subtract_node->output(0)));
}

// Block fails to match if an input is missing
TEST(PatternBlockTest, block_does_not_match_if_input_missing) {
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weights});
    auto add = wrap_type<ov::op::v1::Add>({matmul, bias});

    auto block = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "matmul_add_block");

    Matcher matcher(block);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto matmul_node = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);

    ASSERT_FALSE(matcher.match(matmul_node->output(0)));
}

// Nested Block (Block inside a Block)
TEST(PatternBlockTest, block_nested_blocks) {
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto inner_matmul = wrap_type<ov::op::v0::MatMul>({input, weights});
    auto inner_block =
        std::make_shared<Block>(OutputVector{input, weights}, OutputVector{inner_matmul}, "inner_matmul_block");

    auto add = wrap_type<ov::op::v1::Add>({inner_block, bias});
    auto outer_block = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "outer_block");

    Matcher matcher(outer_block);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto bias_node = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, {0});
    auto matmul_node = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);
    auto add_node = std::make_shared<ov::op::v1::Add>(matmul_node, bias_node);

    ASSERT_TRUE(matcher.match(add_node->output(0)));
}

// Chained Blocks (Block output feeds into another Block)
TEST(PatternBlockTest, block_chained_blocks) {
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto matmul = wrap_type<ov::op::v0::MatMul>({input, weights});
    auto block1 = std::make_shared<Block>(OutputVector{input, weights}, OutputVector{matmul}, "matmul_block");

    auto add = wrap_type<ov::op::v1::Add>({block1, bias});
    auto block2 = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "add_block");

    Matcher matcher(block2);

    auto input_node = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 4});
    auto weights_node = ov::op::v0::Constant::create(element::f32, Shape{4, 4}, {0});
    auto bias_node = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, {0});
    auto matmul_node = std::make_shared<ov::op::v0::MatMul>(input_node, weights_node);
    auto add_node = std::make_shared<ov::op::v1::Add>(matmul_node, bias_node);

    ASSERT_TRUE(matcher.match(add_node->output(0)));
}

TEST(PatternBlockTest, block_registers_and_retrieves_anchors) {
    using ov::op::v0::Constant;
    using ov::op::v0::MatMul;
    using ov::op::v0::Parameter;
    using ov::op::v1::Add;

    // Pattern
    auto input = any_input();
    auto weights = any_input();
    auto bias = any_input();

    auto matmul = wrap_type<MatMul>({input, weights});
    auto add = wrap_type<Add>({matmul, bias});
    auto block = std::make_shared<Block>(OutputVector{input, weights, bias}, OutputVector{add}, "matmul_add_block");
    REGISTER_ANCHORS(block, input, weights, bias, matmul, add);

    // Anchors exist
    const auto& anchors = block->get_registered_anchors();
    EXPECT_EQ(anchors.size(), 5);
    EXPECT_TRUE(anchors.count("input"));
    EXPECT_TRUE(anchors.count("weights"));
    EXPECT_TRUE(anchors.count("bias"));
    EXPECT_TRUE(anchors.count("matmul"));
    EXPECT_TRUE(anchors.count("add"));

    EXPECT_EQ(anchors.at("input"), input);
    EXPECT_EQ(anchors.at("weights"), weights);
    EXPECT_EQ(anchors.at("bias"), bias);
    EXPECT_EQ(anchors.at("matmul"), matmul);
    EXPECT_EQ(anchors.at("add"), add);

    // Real model:
    auto input_node = std::make_shared<Parameter>(element::f32, Shape{1, 4});
    auto weights_node = Constant::create(element::f32, Shape{4, 4}, {1});
    auto bias_node = Constant::create(element::f32, Shape{1, 4}, {2});
    auto matmul_node = std::make_shared<MatMul>(input_node, weights_node);
    auto add_node = std::make_shared<Add>(matmul_node, bias_node);

    Matcher matcher(block);
    ASSERT_TRUE(matcher.match(add_node->output(0)));

    // check that we can use anchors to get matched nodes:
    auto pm = matcher.get_pattern_value_map();
    auto o_input = block->get_anchor("input", pm);
    auto o_weights = block->get_anchor("weights", pm);
    auto o_bias = block->get_anchor("bias", pm);
    auto o_matmul = block->get_anchor("matmul", pm);
    auto o_add = block->get_anchor("add", pm);

    ASSERT_TRUE(o_input.has_value());
    ASSERT_TRUE(o_weights.has_value());
    ASSERT_TRUE(o_bias.has_value());
    ASSERT_TRUE(o_matmul.has_value());
    ASSERT_TRUE(o_add.has_value());

    EXPECT_EQ(o_input.value().get_node_shared_ptr(), input_node);
    EXPECT_EQ(o_weights.value().get_node_shared_ptr(), weights_node);
    EXPECT_EQ(o_bias.value().get_node_shared_ptr(), bias_node);
    EXPECT_EQ(o_matmul.value().get_node_shared_ptr(), matmul_node);
    EXPECT_EQ(o_add.value().get_node_shared_ptr(), add_node);
}
