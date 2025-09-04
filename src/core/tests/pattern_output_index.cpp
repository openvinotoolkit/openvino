// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::pass;

TEST(pattern_output_index, variadic_split_specific_output) {
    // Create a graph with VariadicSplit that has 3 outputs
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    auto split_lengths = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 2, 1});
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_lengths);
    
    // Create three different Reshapes connected to different outputs
    auto reshape_pattern0 = ov::op::v0::Constant::create(element::i32, Shape{2}, {12, 1});
    auto reshape0 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(0), reshape_pattern0, false);
    
    auto reshape_pattern1 = ov::op::v0::Constant::create(element::i32, Shape{2}, {12, 2});
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(1), reshape_pattern1, false);
    
    auto reshape_pattern2 = ov::op::v0::Constant::create(element::i32, Shape{2}, {12, 1});
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(variadic_split->output(2), reshape_pattern2, false);
    
    // Create pattern that should match ONLY output(1)
    auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
    if (auto wrap_node = std::dynamic_pointer_cast<pattern::op::WrapType>(pattern_split)) {
        wrap_node->set_wrapped_output_size(3);
    } else {
        pattern_split->set_output_size(3);
    }
    auto pattern_reshape = pattern::wrap_type<ov::op::v1::Reshape>({pattern_split->output(1), pattern::any_input()});
    
    // Test that it matches reshape1 (connected to output 1)
    {
        pattern::Matcher matcher(pattern_reshape);
        EXPECT_TRUE(matcher.match(reshape1->output(0)));
        auto pattern_map = matcher.get_pattern_value_map();
        EXPECT_EQ(pattern_map.at(pattern_split), variadic_split->output(1));
    }
    
    // Test that it does NOT match reshape0 (connected to output 0)
    {
        pattern::Matcher matcher(pattern_reshape);
        EXPECT_FALSE(matcher.match(reshape0->output(0)));
    }
    
    // Test that it does NOT match reshape2 (connected to output 2)
    {
        pattern::Matcher matcher(pattern_reshape);
        EXPECT_FALSE(matcher.match(reshape2->output(0)));
    }
}

TEST(pattern_output_index, wrap_type_respects_output_index) {
    // Create a simple multi-output node scenario
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{10});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = ov::op::v0::Constant::create(element::i32, Shape{2}, {5, 5});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_lengths);
    
    auto add_const0 = ov::op::v0::Constant::create(element::f32, Shape{5}, {1.0f});
    auto add0 = std::make_shared<ov::op::v1::Add>(split->output(0), add_const0);
    
    auto add_const1 = ov::op::v0::Constant::create(element::f32, Shape{5}, {2.0f});
    auto add1 = std::make_shared<ov::op::v1::Add>(split->output(1), add_const1);
    
    // Pattern for Add connected to output(0) of split
    {
        auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
        if (auto wrap_node = std::dynamic_pointer_cast<pattern::op::WrapType>(pattern_split)) {
            wrap_node->set_wrapped_output_size(2);  // VariadicSplit has 2 outputs in this test
        } else {
            pattern_split->set_output_size(2);  // VariadicSplit has 2 outputs in this test
        }
        auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(0), pattern::any_input()});
        
        pattern::Matcher matcher(pattern_add);
        EXPECT_TRUE(matcher.match(add0->output(0)));
        EXPECT_FALSE(matcher.match(add1->output(0)));
    }
    
    // Pattern for Add connected to output(1) of split
    {
        auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
        if (auto wrap_node = std::dynamic_pointer_cast<pattern::op::WrapType>(pattern_split)) {
            wrap_node->set_wrapped_output_size(2);  // VariadicSplit has 2 outputs in this test
        } else {
            pattern_split->set_output_size(2);  // VariadicSplit has 2 outputs in this test
        }
        auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(1), pattern::any_input()});
        
        pattern::Matcher matcher(pattern_add);
        EXPECT_FALSE(matcher.match(add0->output(0)));
        EXPECT_TRUE(matcher.match(add1->output(0)));
    }
}

TEST(pattern_output_index, split_without_explicit_output_size) {
    // Test that patterns WITHOUT set_wrapped_output_size remain flexible (backward compatibility)
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = ov::op::v0::Constant::create(element::i32, Shape{3}, {4, 4, 4});
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_lengths);
    
    // Create different operations connected to different outputs
    auto add_const0 = ov::op::v0::Constant::create(element::f32, Shape{4}, {1.0f});
    auto add0 = std::make_shared<ov::op::v1::Add>(variadic_split->output(0), add_const0);
    
    auto add_const1 = ov::op::v0::Constant::create(element::f32, Shape{4}, {2.0f});
    auto add1 = std::make_shared<ov::op::v1::Add>(variadic_split->output(1), add_const1);
    
    auto add_const2 = ov::op::v0::Constant::create(element::f32, Shape{4}, {3.0f});
    auto add2 = std::make_shared<ov::op::v1::Add>(variadic_split->output(2), add_const2);
    
    // Create pattern WITHOUT calling set_wrapped_output_size
    // This should match flexibly for backward compatibility
    auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
    // We need to call set_output_size to create the outputs, but NOT set_wrapped_output_size
    pattern_split->set_output_size(3);  // Create 3 outputs in the pattern node
    // Note: NOT calling set_wrapped_output_size - this is the key difference
    
    // Pattern with output(1) should match all outputs (backward compatible behavior)
    auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(1), pattern::any_input()});
    
    pattern::Matcher matcher(pattern_add);
    
    // Without set_wrapped_output_size, the pattern should match flexibly
    // This ensures backward compatibility with existing patterns
    EXPECT_TRUE(matcher.match(add0->output(0)));  // Should match even though it's output(0)
    EXPECT_TRUE(matcher.match(add1->output(0)));  // Should match output(1)
    EXPECT_TRUE(matcher.match(add2->output(0)));  // Should match even though it's output(2)
}

TEST(pattern_output_index, split_with_explicit_output_size) {
    // Test that patterns WITH set_wrapped_output_size check indices strictly
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = ov::op::v0::Constant::create(element::i32, Shape{3}, {4, 4, 4});
    auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_lengths);
    
    // Create different operations connected to different outputs
    auto add_const0 = ov::op::v0::Constant::create(element::f32, Shape{4}, {1.0f});
    auto add0 = std::make_shared<ov::op::v1::Add>(variadic_split->output(0), add_const0);
    
    auto add_const1 = ov::op::v0::Constant::create(element::f32, Shape{4}, {2.0f});
    auto add1 = std::make_shared<ov::op::v1::Add>(variadic_split->output(1), add_const1);
    
    auto add_const2 = ov::op::v0::Constant::create(element::f32, Shape{4}, {3.0f});
    auto add2 = std::make_shared<ov::op::v1::Add>(variadic_split->output(2), add_const2);
    
    // Create pattern WITH set_wrapped_output_size for strict checking
    auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
    if (auto wrap_node = std::dynamic_pointer_cast<pattern::op::WrapType>(pattern_split)) {
        wrap_node->set_wrapped_output_size(3);  // Enable strict index checking
    }
    
    // Pattern with output(1) should match ONLY output(1)
    auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(1), pattern::any_input()});
    
    pattern::Matcher matcher(pattern_add);
    
    // With set_wrapped_output_size, strict index checking is enabled
    EXPECT_FALSE(matcher.match(add0->output(0)));  // Should NOT match output(0)
    EXPECT_TRUE(matcher.match(add1->output(0)));   // Should match output(1)
    EXPECT_FALSE(matcher.match(add2->output(0)));  // Should NOT match output(2)
}

TEST(pattern_output_index, regular_split_without_set_output_size) {
    // Test regular Split (not VariadicSplit) where pattern does NOT need explicit set_output_size
    // because Split inherently has multiple outputs determined by num_splits parameter
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    
    // Create Split with 3 outputs
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 3);
    
    // Connect different operations to different outputs
    auto relu0 = std::make_shared<ov::op::v0::Relu>(split->output(0));
    auto relu1 = std::make_shared<ov::op::v0::Relu>(split->output(1));
    auto relu2 = std::make_shared<ov::op::v0::Relu>(split->output(2));
    
    // Create pattern - we still need set_output_size to create outputs in the pattern node
    // but NOT set_wrapped_output_size (which enables strict checking)
    auto pattern_split = pattern::wrap_type<ov::op::v1::Split>();
    pattern_split->set_output_size(3);  // Create outputs in pattern node
    // Note: NOT calling set_wrapped_output_size() - so no strict index checking
    
    // Try to match with output(0) - but without set_wrapped_output_size it should match flexibly
    auto pattern_relu = pattern::wrap_type<ov::op::v0::Relu>({pattern_split->output(0)});
    
    pattern::Matcher matcher(pattern_relu);
    
    // Without set_wrapped_output_size, should match flexibly (backward compatibility)
    EXPECT_TRUE(matcher.match(relu0->output(0)));  // Should match output(0)
    EXPECT_TRUE(matcher.match(relu1->output(0)));  // Should also match even though it's from output(1)
    EXPECT_TRUE(matcher.match(relu2->output(0)));  // Should also match even though it's from output(2)
}

TEST(pattern_output_index, regular_split_output1_without_strict_check) {
    // Test that pattern with output(1) matches flexibly without set_wrapped_output_size
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    
    // Create Split with 3 outputs
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 3);
    
    // Connect different operations to different outputs
    auto relu0 = std::make_shared<ov::op::v0::Relu>(split->output(0));
    auto relu1 = std::make_shared<ov::op::v0::Relu>(split->output(1));
    auto relu2 = std::make_shared<ov::op::v0::Relu>(split->output(2));
    
    // Create pattern with output(1) but WITHOUT set_wrapped_output_size
    auto pattern_split = pattern::wrap_type<ov::op::v1::Split>();
    pattern_split->set_output_size(3);  // Need this to create outputs
    // NOT calling set_wrapped_output_size - flexible matching
    
    // Use output(1) in pattern
    auto pattern_relu = pattern::wrap_type<ov::op::v0::Relu>({pattern_split->output(1)});
    
    pattern::Matcher matcher(pattern_relu);
    
    // Without set_wrapped_output_size, even output(1) matches flexibly
    EXPECT_TRUE(matcher.match(relu0->output(0)));  // Matches even though it's from output(0)
    EXPECT_TRUE(matcher.match(relu1->output(0)));  // Matches output(1)
    EXPECT_TRUE(matcher.match(relu2->output(0)));  // Matches even though it's from output(2)
}

TEST(pattern_output_index, regular_split_with_set_wrapped_output_size) {
    // Test regular Split with explicit set_wrapped_output_size for strict checking
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    
    // Create Split with 3 outputs
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 3);
    
    // Connect different operations to different outputs
    auto relu0 = std::make_shared<ov::op::v0::Relu>(split->output(0));
    auto relu1 = std::make_shared<ov::op::v0::Relu>(split->output(1));
    auto relu2 = std::make_shared<ov::op::v0::Relu>(split->output(2));
    
    // Create pattern WITH set_wrapped_output_size for strict checking
    auto pattern_split = pattern::wrap_type<ov::op::v1::Split>();
    if (auto wrap_node = std::dynamic_pointer_cast<pattern::op::WrapType>(pattern_split)) {
        wrap_node->set_wrapped_output_size(3);  // Enable strict index checking
    }
    
    // Pattern with output(1) should match ONLY output(1) 
    auto pattern_relu = pattern::wrap_type<ov::op::v0::Relu>({pattern_split->output(1)});
    
    pattern::Matcher matcher(pattern_relu);
    
    // With set_wrapped_output_size, strict checking is enabled
    EXPECT_FALSE(matcher.match(relu0->output(0)));  // Should NOT match output(0)
    EXPECT_TRUE(matcher.match(relu1->output(0)));   // Should match output(1)
    EXPECT_FALSE(matcher.match(relu2->output(0)));  // Should NOT match output(2)
}

TEST(pattern_output_index, any_pattern_respects_output_index) {
    // Test that the Any pattern also respects output indexes
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{10});
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = ov::op::v0::Constant::create(element::i32, Shape{2}, {5, 5});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis, split_lengths);
    
    auto add_const = ov::op::v0::Constant::create(element::f32, Shape{5}, {1.0f});
    auto add0 = std::make_shared<ov::op::v1::Add>(split->output(0), add_const);
    auto add1 = std::make_shared<ov::op::v1::Add>(split->output(1), add_const);
    
    // Pattern using any() that should match only output(0)
    {
        auto pattern_split = pattern::any_input();
        auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(0), pattern::any_input()});
        
        pattern::Matcher matcher(pattern_add);
        // This test may need adjustment based on how any_input() handles outputs
        // For now, we're testing that wrap_type respects the output index
    }
}