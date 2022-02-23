// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, assign_v3_op) {
    NodeBuilder::get_ops().register_factory<opset3::Assign>();
    const auto in = make_shared<op::Parameter>(element::f32, Shape{1});
    const string variable_id = "v0";
    const auto read_value = make_shared<opset3::ReadValue>(in, variable_id);
    const auto assign = make_shared<opset3::Assign>(read_value, variable_id);
    NodeBuilder builder(assign);

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, readvalue_v3_op) {
    NodeBuilder::get_ops().register_factory<opset3::ReadValue>();
    const auto in = make_shared<op::Parameter>(element::f32, Shape{1});
    const string variable_id = "v0";
    const auto read_value = make_shared<opset3::ReadValue>(in, variable_id);
    NodeBuilder builder(read_value);

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

/*
TEST(attributes, assign_readvalue_v6_op) {
    NodeBuilder::get_ops().register_factory<opset6::ReadValue>();
    auto input = make_shared<op::Parameter>(element::f16, Shape{4, 3, 2, 1});
    auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = make_shared<opset6::ReadValue>(input, variable);
    // auto assign = make_shared<opset6::Assign>(read_value, variable);
    NodeBuilder builder(read_value);

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(type_prop, read_value_deduce) {
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<opset5::ReadValue>(input, "variable_id");

    ASSERT_EQ(read_value->get_element_type(), element::f32);
    ASSERT_EQ(read_value->get_shape(), (Shape{1, 2, 64, 64}));
}
TEST(type_prop, assign_deduce) {
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 64, 64});
    auto read_value = make_shared<opset5::ReadValue>(input, "variable_id");
    auto assign = make_shared<opset5::Assign>(read_value, "variable_id");

    ASSERT_EQ(assign->get_element_type(), element::f32);
    ASSERT_EQ(assign->get_shape(), (Shape{1, 2, 64, 64}));
}

TEST(type_prop, assign_read_value_new_shape) {
    auto input = make_shared<op::Parameter>(element::f16, Shape{4, 3, 2, 1});

    auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = make_shared<opset6::ReadValue>(input, variable);
    auto assign = make_shared<opset6::Assign>(read_value, variable);

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_shape(), (Shape{4, 3, 2, 1}));

    auto f = std::make_shared<Function>(ResultVector{}, SinkVector{assign}, ParameterVector{input});

    input->set_partial_shape({3, {4, 5}, 8});
    f->validate_nodes_and_infer_types();

    ASSERT_EQ(assign->get_element_type(), element::f16);
    ASSERT_EQ(assign->get_output_partial_shape(0), (PartialShape{3, {4, 5}, 8}));
    ASSERT_EQ(variable->get_info().data_type, element::f16);
    ASSERT_EQ(variable->get_info().data_shape, (PartialShape{3, {4, 5}, 8}));
}
*/