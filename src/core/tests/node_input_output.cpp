// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace std;

TEST(node_input_output, input_create) {
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
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

    EXPECT_THROW(add->input(2), std::out_of_range);
}

TEST(node_input_output, input_create_const) {
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
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

    EXPECT_THROW(add->input(2), std::out_of_range);
}

TEST(node_input_output, output_create) {
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
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

    EXPECT_THROW(add->output(1), std::out_of_range);
}

TEST(node_input_output, output_create_const) {
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto add = make_shared<const op::v1::Add>(x, y);

    auto add_out_0 = add->output(0);

    EXPECT_EQ(add_out_0.get_names().size(), 0);
    EXPECT_EQ(add_out_0.get_node(), add.get());
    EXPECT_EQ(add_out_0.get_index(), 0);
    EXPECT_EQ(add_out_0.get_element_type(), element::f32);
    EXPECT_EQ(add_out_0.get_shape(), (Shape{1, 2, 3, 4}));
    EXPECT_TRUE(add_out_0.get_partial_shape().same_scheme(PartialShape{1, 2, 3, 4}));

    EXPECT_THROW(add->output(1), std::out_of_range);
}

TEST(node_input_output, output_rt_info) {
    auto x = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
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
    auto x = make_shared<op::Parameter>(element::f32, Shape{1});
    auto y = make_shared<op::Parameter>(element::f32, Shape{2});
    auto z = make_shared<op::Parameter>(element::f32, Shape{3});

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

TEST(node_input_output, output_name_for_v9_ops) {
    const std::string ref_input_name = "input_tensor_name";
    const std::string ref_output_name = "output_tensor_name";
    auto param = make_shared<op::v9::Parameter>(element::f32, Shape{1, 2, 3, 4}, ref_input_name);
    std::unordered_set<std::string> ref_val = {"1", "2", "3"};
    param->get_output_tensor(0).set_names(ref_val);
    EXPECT_EQ(ref_val, param->get_output_tensor(0).get_names());

    auto result = make_shared<op::v9::Result>(param, ref_output_name);
    auto model = make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});
    model->validate_nodes_and_infer_types();

    ref_val.insert(ref_input_name);
    ref_val.insert(ref_output_name);
    EXPECT_EQ(ref_val, param->get_output_tensor(0).get_names());
    EXPECT_EQ(ref_input_name, model->input().get_name());
    EXPECT_EQ(ref_output_name, model->output().get_name());
}
