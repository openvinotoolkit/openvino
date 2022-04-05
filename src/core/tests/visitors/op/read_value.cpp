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

TEST(attributes, readvalue_v6_op) {
    NodeBuilder::get_ops().register_factory<opset6::ReadValue>();
    const auto in = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    const auto read_value = make_shared<opset6::ReadValue>(in, variable);
    NodeBuilder builder(read_value);

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
