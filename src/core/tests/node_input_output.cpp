// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <set>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/convert.hpp"

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
    ASSERT_EQ(add1_targets.size(), 3); // mul has 2 inputs + relu has 1 input
    
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
        EXPECT_EQ(input.get_source_output().get_node(), add2.get())
            << "All target inputs should point to add2";
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
    ASSERT_EQ(add->output(0).get_target_inputs().size(), 1); // convert2
    ASSERT_EQ(convert2->output(0).get_target_inputs().size(), 1); // relu
    
    // Replace convert2's output with add's output (removing unnecessary conversion)
    convert2->output(0).replace(add->output(0));
    
    // Check that add has relu as target and NOT convert2
    auto add_targets = add->output(0).get_target_inputs();
    set<Node*> target_nodes;
    for (const auto& input : add_targets) {
        target_nodes.insert(input.get_node());
    }
    
    EXPECT_TRUE(target_nodes.count(relu.get()) > 0) 
        << "add should have relu as a target after replace";
    EXPECT_FALSE(target_nodes.count(convert2.get()) > 0) 
        << "add should NOT have convert2 as a target after replace (this was the bug)";
    
    // convert2 should have no targets
    EXPECT_EQ(convert2->output(0).get_target_inputs().size(), 0) 
        << "convert2 should have no targets after replace";
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
    ASSERT_EQ(add1->output(0).get_target_inputs().size(), 5); // relu1, relu2, mul(2 inputs), convert
    
    // Create replacement node
    auto add2 = make_shared<op::v1::Add>(param, param);
    
    // Perform replacement
    add1->output(0).replace(add2->output(0));
    
    // Verify all connections moved correctly regardless of iteration order
    auto add2_targets = add2->output(0).get_target_inputs();
    EXPECT_EQ(add2_targets.size(), 5) << "All connections should move to add2";
    
    // Verify specific connections
    set<Node*> target_nodes;
    for (const auto& input : add2_targets) {
        target_nodes.insert(input.get_node());
    }
    
    EXPECT_TRUE(target_nodes.count(relu1.get()) > 0) << "relu1 should connect to add2";
    EXPECT_TRUE(target_nodes.count(relu2.get()) > 0) << "relu2 should connect to add2";
    EXPECT_TRUE(target_nodes.count(mul.get()) > 0) << "mul should connect to add2";
    EXPECT_TRUE(target_nodes.count(convert.get()) > 0) << "convert should connect to add2";
    
    // Original node should have no connections
    EXPECT_EQ(add1->output(0).get_target_inputs().size(), 0) 
        << "add1 should have no connections after replace";
    
    // Verify mul has both inputs from add2
    EXPECT_EQ(mul->input_value(0).get_node(), add2.get());
    EXPECT_EQ(mul->input_value(1).get_node(), add2.get());
}
