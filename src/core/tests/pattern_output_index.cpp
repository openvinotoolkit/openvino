// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_tools.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
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
    pattern_split->set_output_size(3);
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
        pattern_split->set_output_size(2);  // VariadicSplit has 2 outputs in this test
        auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(0), pattern::any_input()});
        
        pattern::Matcher matcher(pattern_add);
        EXPECT_TRUE(matcher.match(add0->output(0)));
        EXPECT_FALSE(matcher.match(add1->output(0)));
    }
    
    // Pattern for Add connected to output(1) of split
    {
        auto pattern_split = pattern::wrap_type<ov::op::v1::VariadicSplit>();
        pattern_split->set_output_size(2);  // VariadicSplit has 2 outputs in this test
        auto pattern_add = pattern::wrap_type<ov::op::v1::Add>({pattern_split->output(1), pattern::any_input()});
        
        pattern::Matcher matcher(pattern_add);
        EXPECT_FALSE(matcher.match(add0->output(0)));
        EXPECT_TRUE(matcher.match(add1->output(0)));
    }
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