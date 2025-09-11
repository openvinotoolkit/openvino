// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

using namespace ov;
using namespace std;

TEST(node_input_output, input_create) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(x, y);

    auto add_in_0 = add->input(0);
    auto add_in_1 = add->input(1);

    EXPECT_EQ(add_in_0.get_node(), add.get());
    EXPECT_EQ(add_in_0.get_index(), 0);
    EXPECT_EQ(add_in_0.get_element_type(), element::f32);
    EXPECT_EQ(add_in_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_0.get_source_output(), Output<Node>(x, 0));

    EXPECT_EQ(add_in_1.get_node(), add.get());
    EXPECT_EQ(add_in_1.get_index(), 1);
    EXPECT_EQ(add_in_1.get_element_type(), element::f32);
    EXPECT_EQ(add_in_1.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_1.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_1.get_source_output(), Output<Node>(y, 0));

    EXPECT_THROW(add->input(2), ov::Exception);
}

TEST(node_input_output, input_create_const) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<const op::v1::Add>(x, y);

    auto add_in_0 = add->input(0);
    auto add_in_1 = add->input(1);

    EXPECT_EQ(add_in_0.get_node(), add.get());
    EXPECT_EQ(add_in_0.get_index(), 0);
    EXPECT_EQ(add_in_0.get_element_type(), element::f32);
    EXPECT_EQ(add_in_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_0.get_source_output(), Output<Node>(x, 0));

    EXPECT_EQ(add_in_1.get_node(), add.get());
    EXPECT_EQ(add_in_1.get_index(), 1);
    EXPECT_EQ(add_in_1.get_element_type(), element::f32);
    EXPECT_EQ(add_in_1.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_in_1.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));
    EXPECT_EQ(add_in_1.get_source_output(), Output<Node>(y, 0));

    EXPECT_THROW(add->input(2), ov::Exception);
}

TEST(node_input_output, output_create) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(x, y);

    auto add_out_0 = add->output(0);
    add_out_0.set_names({"a", "b"});
    EXPECT_EQ(add_out_0.get_names(), std::unordered_set<std::string>({"a", "b"}));
    EXPECT_EQ(add_out_0.get_any_name(), "a");
    add_out_0.add_names({"c", "d"});
    EXPECT_EQ(add_out_0.get_names(), std::unordered_set<std::string>({"a", "b", "c", "d"}));
    EXPECT_EQ(add_out_0.get_any_name(), "a");

    EXPECT_EQ(add_out_0.get_node(), add.get());
    EXPECT_EQ(add_out_0.get_index(), 0);
    EXPECT_EQ(add_out_0.get_element_type(), element::f32);
    EXPECT_EQ(add_out_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_out_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));

    EXPECT_THROW(add->output(1), ov::Exception);
}

TEST(node_input_output, output_create_const) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<const op::v1::Add>(x, y);

    auto add_out_0 = add->output(0);

    EXPECT_EQ(add_out_0.get_names().size(), 0);
    EXPECT_EQ(add_out_0.get_node(), add.get());
    EXPECT_EQ(add_out_0.get_index(), 0);
    EXPECT_EQ(add_out_0.get_element_type(), element::f32);
    EXPECT_EQ(add_out_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_out_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));

    EXPECT_THROW(add->output(1), ov::Exception);
}

TEST(node_input_output, output_rt_info) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(x, y);
    auto add_const = make_shared<const op::v1::Add>(x, y);

    Output<Node> output = add->output(0);
    Output<const Node> output_const = add_const->output(0);

    auto& rt = output.get_rt_info();
    rt["test"] = nullptr;
    EXPECT_TRUE(output.get_rt_info().count("test"));
    EXPECT_TRUE(output.get_tensor_ptr()->get_rt_info().count("test"));
    EXPECT_TRUE(output_const.get_rt_info().empty());
}

TEST(node_input_output, input_set_argument) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto z = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    auto add = make_shared<op::v1::Add>(x, y);

    EXPECT_EQ(add->get_input_size(), 2);
    EXPECT_EQ(add->input(0).get_shape(), Shape{1});
    EXPECT_EQ(add->input(1).get_shape(), Shape{2});

    add->set_argument(1, z);

    EXPECT_EQ(add->get_input_size(), 2);
    EXPECT_EQ(add->input(0).get_shape(), Shape{1});
    EXPECT_EQ(add->input(1).get_shape(), Shape{3});

    add->set_arguments(NodeVector{z, x});

    EXPECT_EQ(add->get_input_size(), 2);
    EXPECT_EQ(add->input(0).get_shape(), Shape{3});
    EXPECT_EQ(add->input(1).get_shape(), Shape{1});
}

TEST(node_input_output, create_wrong_input_output) {
    EXPECT_THROW(ov::Output<ov::Node>(nullptr, 0), ov::Exception);
    EXPECT_THROW(ov::Output<const ov::Node>(nullptr, 0), ov::Exception);
    EXPECT_THROW(ov::Input<ov::Node>(nullptr, 0), ov::Exception);
    EXPECT_THROW(ov::Input<const ov::Node>(nullptr, 0), ov::Exception);
}

TEST(node_input_output, output_replace_removes_all_connections) {
    // Test for issue #107966: Output<Node>::replace doesn't properly remove existing connections
    // Create initial graph: param -> add1 -> mul, relu
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto add1 = make_shared<op::v1::Add>(param, param);
    auto mul = make_shared<op::v1::Multiply>(add1, add1);
    auto relu = make_shared<op::v0::Relu>(add1);

    // Verify initial connections
    auto add1_targets = add1->output(0).get_target_inputs();
    ASSERT_EQ(add1_targets.size(), 3);  // mul has 2 inputs + relu has 1 input

    // Create replacement node
    auto add2 = make_shared<op::v1::Add>(param, param);

    // Replace add1's output with add2's output
    add1->output(0).replace(add2->output(0));

    // Verify all connections moved to add2
    auto add2_targets = add2->output(0).get_target_inputs();
    EXPECT_EQ(add2_targets.size(), 3) << "add2 should have all connections from add1";

    // Verify add1 has no connections left
    auto add1_targets_after = add1->output(0).get_target_inputs();
    EXPECT_EQ(add1_targets_after.size(), 0) << "add1 should have no connections after replace";

    // Verify all connections point to add2
    for (const auto& input : add2_targets) {
        EXPECT_EQ(input.get_source_output().get_node(), add2.get()) << "All target inputs should point to add2";
    }
}

TEST(node_input_output, output_replace_with_existing_connection) {
    // Test the specific scenario from the patch where replace() doesn't work correctly
    // when the replacement node already has connections to the same targets
    // Issue #107966
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto convert1 = make_shared<op::v0::Convert>(param, element::bf16);
    auto add = make_shared<op::v1::Add>(convert1, convert1);
    auto convert2 = make_shared<op::v0::Convert>(add, element::f32);
    auto relu = make_shared<op::v0::Relu>(convert2);

    // Verify initial state
    ASSERT_EQ(add->output(0).get_target_inputs().size(), 1);       // convert2
    ASSERT_EQ(convert2->output(0).get_target_inputs().size(), 1);  // relu

    // Replace convert2's output with add's output (removing unnecessary conversion)
    convert2->output(0).replace(add->output(0));

    // Check that add has relu as target and NOT convert2
    auto add_targets = add->output(0).get_target_inputs();
    EXPECT_EQ(add_targets.size(), 1) << "add should have exactly one target after replace";
    EXPECT_EQ(add_targets.begin()->get_node(), relu.get()) << "add's only target should be relu";

    // convert2 should have no targets
    EXPECT_EQ(convert2->output(0).get_target_inputs().size(), 0) << "convert2 should have no targets after replace";
}

TEST(node_input_output, output_replace_order_independence) {
    // Test that the order of processing target_inputs in replace() doesn't affect the result
    // This test checks if the implementation is robust against different iteration orders

    // Create a complex graph with multiple connections
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
    auto add1 = make_shared<op::v1::Add>(param, param);

    // Create multiple consumers with different node types
    auto relu1 = make_shared<op::v0::Relu>(add1);
    auto relu2 = make_shared<op::v0::Relu>(add1);
    auto mul = make_shared<op::v1::Multiply>(add1, add1);
    auto convert = make_shared<op::v0::Convert>(add1, element::bf16);

    // Store initial connections count
    ASSERT_EQ(add1->output(0).get_target_inputs().size(), 5);  // relu1, relu2, mul(2 inputs), convert

    // Create replacement node
    auto add2 = make_shared<op::v1::Add>(param, param);

    // Perform replacement
    add1->output(0).replace(add2->output(0));

    // Verify all connections moved correctly regardless of iteration order
    auto add2_targets = add2->output(0).get_target_inputs();
    EXPECT_EQ(add2_targets.size(), 5) << "All connections should move to add2";

    // Verify specific connections - mul appears twice (2 inputs)
    int relu1_count = 0, relu2_count = 0, mul_count = 0, convert_count = 0;
    for (const auto& input : add2_targets) {
        if (input.get_node() == relu1.get()) relu1_count++;
        if (input.get_node() == relu2.get()) relu2_count++;
        if (input.get_node() == mul.get()) mul_count++;
        if (input.get_node() == convert.get()) convert_count++;
    }
    EXPECT_EQ(relu1_count, 1) << "relu1 should connect to add2 once";
    EXPECT_EQ(relu2_count, 1) << "relu2 should connect to add2 once";
    EXPECT_EQ(mul_count, 2) << "mul should connect to add2 twice (both inputs)";
    EXPECT_EQ(convert_count, 1) << "convert should connect to add2 once";

    // Original node should have no connections
    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0) << "add1 should have no connections after replace";

    // Verify mul has both inputs from add2
    EXPECT_EQ(mul->input_value(0).get_node(), add2.get());
    EXPECT_EQ(mul->input_value(1).get_node(), add2.get());
}

TEST(node_input_output, output_replace_self_loop) {
    // Corner case: replacing output with itself should be no-op
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add = make_shared<op::v1::Add>(param, param);
    auto relu = make_shared<op::v0::Relu>(add);

    // Count connections before
    auto initial_targets = add->output(0).get_target_inputs();
    ASSERT_EQ(initial_targets.size(), 1);  // relu

    // Replace output with itself
    add->output(0).replace(add->output(0));

    // Should be unchanged
    auto final_targets = add->output(0).get_target_inputs();
    EXPECT_EQ(final_targets.size(), 1) << "Self-replacement should preserve connections";
    EXPECT_EQ(relu->input_value(0).get_node(), add.get()) << "Relu should still connect to add";
}

TEST(node_input_output, output_replace_chain_of_converts) {
    // Corner case: chain of conversions A -> B -> C -> D, replace C with A
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto conv_a = make_shared<op::v0::Convert>(param, element::bf16);
    auto conv_b = make_shared<op::v0::Convert>(conv_a, element::f16);
    auto conv_c = make_shared<op::v0::Convert>(conv_b, element::i32);
    auto conv_d = make_shared<op::v0::Convert>(conv_c, element::f32);
    auto relu = make_shared<op::v0::Relu>(conv_d);

    // Initial state check
    ASSERT_EQ(conv_a->output(0).get_target_inputs().size(), 1);  // conv_b
    ASSERT_EQ(conv_c->output(0).get_target_inputs().size(), 1);  // conv_d

    // Replace conv_c's output with conv_a's output
    conv_c->output(0).replace(conv_a->output(0));

    // Check result
    auto conv_a_targets = conv_a->output(0).get_target_inputs();
    EXPECT_EQ(conv_a_targets.size(), 2) << "conv_a should have conv_b and conv_d as targets";

    // Verify both expected targets are present
    bool has_conv_b = false, has_conv_d = false;
    for (const auto& input : conv_a_targets) {
        if (input.get_node() == conv_b.get()) has_conv_b = true;
        if (input.get_node() == conv_d.get()) has_conv_d = true;
    }
    EXPECT_TRUE(has_conv_b) << "conv_b should still connect to conv_a";
    EXPECT_TRUE(has_conv_d) << "conv_d should now connect to conv_a";

    EXPECT_EQ(conv_c->output(0).get_target_inputs().size(), 0) << "conv_c should have no targets";
}

TEST(node_input_output, output_replace_multiple_outputs) {
    // Corner case: node with multiple outputs
    auto param1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto param2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});

    // Create a node with 2 outputs (using Split for example would be better, but Add works for test)
    auto add1 = make_shared<op::v1::Add>(param1, param2);
    auto add2 = make_shared<op::v1::Add>(param1, param2);

    // Connect different consumers to each output
    auto relu1 = make_shared<op::v0::Relu>(add1);
    auto relu2 = make_shared<op::v0::Relu>(add1);
    auto conv = make_shared<op::v0::Convert>(add2, element::bf16);

    // Replace add1's output with add2's output
    add1->output(0).replace(add2->output(0));

    // Check that connections moved correctly
    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0) << "add1 should have no targets";

    auto add2_targets = add2->output(0).get_target_inputs();
    EXPECT_EQ(add2_targets.size(), 3) << "add2 should have relu1, relu2, and conv as targets";
    
    // Verify all expected targets are present
    bool has_relu1 = false, has_relu2 = false, has_conv = false;
    for (const auto& input : add2_targets) {
        if (input.get_node() == relu1.get()) has_relu1 = true;
        if (input.get_node() == relu2.get()) has_relu2 = true;
        if (input.get_node() == conv.get()) has_conv = true;
    }
    EXPECT_TRUE(has_relu1) << "relu1 should connect to add2";
    EXPECT_TRUE(has_relu2) << "relu2 should connect to add2";
    EXPECT_TRUE(has_conv) << "conv should still connect to add2";
}

TEST(node_input_output, output_replace_bidirectional_connection) {
    // Corner case: A -> B and B -> A (though this shouldn't normally happen in a DAG)
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add1 = make_shared<op::v1::Add>(param, param);
    auto relu = make_shared<op::v0::Relu>(add1);
    auto add2 = make_shared<op::v1::Add>(relu, param);

    // add2 uses relu which uses add1
    // Now if we try to make add1 use add2's output (creating a cycle in theory)
    // This test ensures replace() handles it gracefully

    ASSERT_EQ(add1->output(0).get_target_inputs().size(), 1);  // relu
    ASSERT_EQ(add2->output(0).get_target_inputs().size(), 0);  // no consumers yet

    // Add a consumer to add2 that is add1 (simulating the problematic case)
    auto mul = make_shared<op::v1::Multiply>(add2, add1);

    // Now replace add1's output with add2's output
    add1->output(0).replace(add2->output(0));

    // Check results
    auto add2_targets = add2->output(0).get_target_inputs();
    // Note: mul has 2 inputs from add2 (first operand) and add1 (second operand)
    // After replace, mul's second input should also point to add2
    EXPECT_EQ(add2_targets.size(), 3) << "add2 should have relu + mul(2 inputs) as targets";

    // Count connections to each node
    int relu_count = 0, mul_count = 0;
    for (const auto& input : add2_targets) {
        if (input.get_node() == relu.get()) relu_count++;
        if (input.get_node() == mul.get()) mul_count++;
    }
    EXPECT_EQ(relu_count, 1) << "relu should have exactly 1 connection from add2";
    EXPECT_EQ(mul_count, 2) << "mul should have exactly 2 connections from add2";

    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0) << "add1 should have no targets";
}

TEST(node_input_output, output_replace_empty_targets) {
    // Corner case: replacing output that has no targets
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add1 = make_shared<op::v1::Add>(param, param);
    auto add2 = make_shared<op::v1::Add>(param, param);
    auto relu = make_shared<op::v0::Relu>(add2);

    // add1 has no targets, add2 has relu
    ASSERT_EQ(add1->output(0).get_target_inputs().size(), 0);
    ASSERT_EQ(add2->output(0).get_target_inputs().size(), 1);

    // Replace add1's output (no targets) with add2's output (has targets)
    add1->output(0).replace(add2->output(0));

    // Nothing should change since add1 had no targets
    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0);
    EXPECT_EQ(add2->output(0).get_target_inputs().size(), 1);
    EXPECT_EQ(relu->input_value(0).get_node(), add2.get());
}

TEST(node_input_output, output_replace_with_parameter) {
    // Corner case: replacing with parameter output
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add = make_shared<op::v1::Add>(param, param);
    auto relu = make_shared<op::v0::Relu>(add);

    // Initial state: param has 2 connections to add
    ASSERT_EQ(param->output(0).get_target_inputs().size(), 2);

    // Replace add's output with parameter's output
    add->output(0).replace(param->output(0));

    // Check that relu now connects to parameter
    EXPECT_EQ(relu->input_value(0).get_node(), param.get()) << "Relu should connect directly to parameter";
    EXPECT_EQ(add->output(0).get_target_inputs().size(), 0) << "Add should have no targets";

    // After replacement, param should have only relu as target
    // because our fix removes cyclic connections
    auto param_targets = param->output(0).get_target_inputs();
    EXPECT_EQ(param_targets.size(), 1) << "Parameter should have only relu as target";
    ASSERT_EQ(param_targets.begin()->get_node(), relu.get()) << "The only target should be relu";
}

TEST(node_input_output, output_replace_cascade) {
    // Corner case: cascade of replacements
    auto param = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3});
    auto add1 = make_shared<op::v1::Add>(param, param);
    auto add2 = make_shared<op::v1::Add>(add1, param);
    auto add3 = make_shared<op::v1::Add>(add2, param);
    auto relu = make_shared<op::v0::Relu>(add3);

    // Replace in cascade: add3 -> add2 -> add1 -> param
    add3->output(0).replace(add2->output(0));
    EXPECT_EQ(relu->input_value(0).get_node(), add2.get());

    add2->output(0).replace(add1->output(0));
    EXPECT_EQ(relu->input_value(0).get_node(), add1.get());

    add1->output(0).replace(param->output(0));
    EXPECT_EQ(relu->input_value(0).get_node(), param.get());

    // Check all intermediate nodes have no targets
    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0);
    EXPECT_EQ(add2->output(0).get_target_inputs().size(), 0);
    EXPECT_EQ(add3->output(0).get_target_inputs().size(), 0);
}
