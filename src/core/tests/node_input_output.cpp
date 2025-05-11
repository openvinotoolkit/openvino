// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"

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
