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

namespace ov {
namespace test {

using namespace ov;
using namespace ov::pass;

TEST(pattern_output_index, variadic_split_strict_output_matching) {
    // Test that patterns with specific output indices match only those outputs
    // With the new behavior, strict index checking is always enabled for multi-output nodes
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {1});
    auto split_lengths = op::v0::Constant::create(element::i32, Shape{3}, {1, 2, 1});
    auto variadic_split = std::make_shared<op::v1::VariadicSplit>(input, axis, split_lengths);

    // Create operations connected to different outputs
    auto reshape_pattern0 = op::v0::Constant::create(element::i32, Shape{2}, {12, 1});
    auto reshape0 = std::make_shared<op::v1::Reshape>(variadic_split->output(0), reshape_pattern0, false);

    auto reshape_pattern1 = op::v0::Constant::create(element::i32, Shape{2}, {12, 2});
    auto reshape1 = std::make_shared<op::v1::Reshape>(variadic_split->output(1), reshape_pattern1, false);

    auto reshape_pattern2 = op::v0::Constant::create(element::i32, Shape{2}, {12, 1});
    auto reshape2 = std::make_shared<op::v1::Reshape>(variadic_split->output(2), reshape_pattern2, false);

    // Test pattern matching with output(0)
    {
        auto pattern_split = pattern::wrap_type<op::v1::VariadicSplit>();
        pattern_split->set_output_size(3);  // Pattern node needs to know it has 3 outputs
        auto pattern_reshape = pattern::wrap_type<op::v1::Reshape>({pattern_split->output(0), pattern::any_input()});

        pattern::Matcher matcher(pattern_reshape);
        EXPECT_TRUE(matcher.match(reshape0->output(0)));   // Should match output(0)
        EXPECT_FALSE(matcher.match(reshape1->output(0)));  // Should NOT match output(1)
        EXPECT_FALSE(matcher.match(reshape2->output(0)));  // Should NOT match output(2)
    }

    // Test pattern matching with output(1)
    {
        auto pattern_split = pattern::wrap_type<op::v1::VariadicSplit>();
        pattern_split->set_output_size(3);  // Pattern node needs to know it has 3 outputs
        auto pattern_reshape = pattern::wrap_type<op::v1::Reshape>({pattern_split->output(1), pattern::any_input()});

        pattern::Matcher matcher(pattern_reshape);
        EXPECT_FALSE(matcher.match(reshape0->output(0)));  // Should NOT match output(0)
        EXPECT_TRUE(matcher.match(reshape1->output(0)));   // Should match output(1)
        EXPECT_FALSE(matcher.match(reshape2->output(0)));  // Should NOT match output(2)
    }

    // Test pattern matching with output(2)
    {
        auto pattern_split = pattern::wrap_type<op::v1::VariadicSplit>();
        pattern_split->set_output_size(3);  // Pattern node needs to know it has 3 outputs
        auto pattern_reshape = pattern::wrap_type<op::v1::Reshape>({pattern_split->output(2), pattern::any_input()});

        pattern::Matcher matcher(pattern_reshape);
        EXPECT_FALSE(matcher.match(reshape0->output(0)));  // Should NOT match output(0)
        EXPECT_FALSE(matcher.match(reshape1->output(0)));  // Should NOT match output(1)
        EXPECT_TRUE(matcher.match(reshape2->output(0)));   // Should match output(2)
    }
}

TEST(pattern_output_index, split_strict_output_matching) {
    // Test regular Split with strict output index checking
    // Strict checking is now always enabled for multi-output nodes
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{12});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split = std::make_shared<op::v1::Split>(input, axis, 3);

    // Create operations connected to different outputs
    auto add_const0 = op::v0::Constant::create(element::f32, Shape{4}, {1.0f});
    auto add0 = std::make_shared<op::v1::Add>(split->output(0), add_const0);

    auto add_const1 = op::v0::Constant::create(element::f32, Shape{4}, {2.0f});
    auto add1 = std::make_shared<op::v1::Add>(split->output(1), add_const1);

    auto add_const2 = op::v0::Constant::create(element::f32, Shape{4}, {3.0f});
    auto add2 = std::make_shared<op::v1::Add>(split->output(2), add_const2);

    // Test pattern with output(1)
    auto pattern_split = pattern::wrap_type<op::v1::Split>();
    pattern_split->set_output_size(3);  // Pattern node needs to know it has 3 outputs
    auto pattern_add = pattern::wrap_type<op::v1::Add>({pattern_split->output(1), pattern::any_input()});

    pattern::Matcher matcher(pattern_add);
    EXPECT_FALSE(matcher.match(add0->output(0)));  // Should NOT match - different output index
    EXPECT_TRUE(matcher.match(add1->output(0)));   // Should match - same output index (1)
    EXPECT_FALSE(matcher.match(add2->output(0)));  // Should NOT match - different output index
}

TEST(pattern_output_index, multi_output_node_strict_index_checking) {
    // Test that all multi-output nodes have strict index checking automatically enabled
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = op::v0::Constant::create(element::i32, Shape{2}, {3, 7});
    auto split = std::make_shared<op::v1::VariadicSplit>(input, axis, split_lengths);

    // Create different operations on different outputs
    auto const0 = op::v0::Constant::create(element::f32, Shape{3}, {1.0});
    auto add0 = std::make_shared<op::v1::Add>(split->output(0), const0);  // Connected to output 0

    auto const1 = op::v0::Constant::create(element::f32, Shape{7}, {2.0});
    auto add1 = std::make_shared<op::v1::Add>(split->output(1), const1);  // Connected to output 1

    // Pattern expecting output(1) - strict checking is automatic now
    auto pattern_split = pattern::wrap_type<op::v1::VariadicSplit>();
    pattern_split->set_output_size(2);  // Pattern node needs to know it has 2 outputs
    auto pattern_add = pattern::wrap_type<op::v1::Add>({pattern_split->output(1), pattern::any_input()});

    pattern::Matcher matcher(pattern_add);

    // Strict index checking is always enabled for multi-output nodes
    EXPECT_FALSE(matcher.match(add0->output(0)));  // Should NOT match - add0 is on output(0)
    EXPECT_TRUE(matcher.match(add1->output(0)));   // Should match - add1 is on output(1)
}

TEST(pattern_output_index, pattern_or_for_flexible_matching) {
    // Test using pattern::op::Or for flexible output matching when needed
    // This is the recommended approach when you want to match any output
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{12, 4});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split = std::make_shared<op::v1::Split>(input, axis, 3);

    auto relu0 = std::make_shared<op::v0::Relu>(split->output(0));
    auto relu1 = std::make_shared<op::v0::Relu>(split->output(1));
    auto relu2 = std::make_shared<op::v0::Relu>(split->output(2));

    // Pattern that can match any output using Or - this is how to achieve flexible matching
    auto pattern_split = pattern::wrap_type<op::v1::Split>();
    pattern_split->set_output_size(3);  // Pattern node needs to know it has 3 outputs
    auto any_output = std::make_shared<pattern::op::Or>(pattern_split->outputs());
    auto pattern_relu = pattern::wrap_type<op::v0::Relu>({any_output});

    pattern::Matcher matcher(pattern_relu);

    // Should match all outputs when using Or
    EXPECT_TRUE(matcher.match(relu0->output(0)));
    EXPECT_TRUE(matcher.match(relu1->output(0)));
    EXPECT_TRUE(matcher.match(relu2->output(0)));
}

TEST(pattern_output_index, single_output_nodes_unaffected) {
    // Test that single-output nodes are unaffected by strict index checking
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{10});
    auto relu = std::make_shared<op::v0::Relu>(input);
    auto add_const = op::v0::Constant::create(element::f32, Shape{10}, {1.0f});
    auto add = std::make_shared<op::v1::Add>(relu, add_const);

    // Pattern for single-output nodes
    auto pattern_relu = pattern::wrap_type<op::v0::Relu>();
    auto pattern_add = pattern::wrap_type<op::v1::Add>({pattern_relu, pattern::any_input()});

    pattern::Matcher matcher(pattern_add);
    EXPECT_TRUE(matcher.match(add->output(0)));  // Should match normally
}

TEST(pattern_output_index, verify_index_mismatch_prevents_matching) {
    // Comprehensive test to verify that index mismatches prevent matching
    // With the new behavior, strict checking is automatic
    const size_t num_outputs = 4;
    auto input = std::make_shared<op::v0::Parameter>(element::f32, Shape{12});
    auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
    auto split_lengths = op::v0::Constant::create(element::i32, Shape{num_outputs}, {3, 3, 3, 3});
    auto split = std::make_shared<op::v1::VariadicSplit>(input, axis, split_lengths);

    // Create operations on all outputs
    std::vector<std::shared_ptr<op::v0::Relu>> relus;
    for (size_t i = 0; i < num_outputs; ++i) {
        relus.push_back(std::make_shared<op::v0::Relu>(split->output(i)));
    }

    // Test each pattern output against all graph outputs
    for (size_t pattern_idx = 0; pattern_idx < num_outputs; ++pattern_idx) {
        auto pattern_split = pattern::wrap_type<op::v1::VariadicSplit>();
        pattern_split->set_output_size(num_outputs);  // Pattern node needs to know it has num_outputs outputs
        auto pattern_relu = pattern::wrap_type<op::v0::Relu>({pattern_split->output(pattern_idx)});

        pattern::Matcher matcher(pattern_relu);

        for (size_t graph_idx = 0; graph_idx < num_outputs; ++graph_idx) {
            if (pattern_idx == graph_idx) {
                EXPECT_TRUE(matcher.match(relus[graph_idx]->output(0)))
                    << "Pattern output(" << pattern_idx << ") should match graph output(" << graph_idx << ")";
            } else {
                EXPECT_FALSE(matcher.match(relus[graph_idx]->output(0)))
                    << "Pattern output(" << pattern_idx << ") should NOT match graph output(" << graph_idx << ")";
            }
        }
    }
}

}  // namespace test
}  // namespace ov