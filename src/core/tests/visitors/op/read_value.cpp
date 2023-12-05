// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/read_value.hpp"

#include <gtest/gtest.h>

#include "openvino/op/util/variable.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, readvalue_v3_op) {
    NodeBuilder::get_ops().register_factory<ov::op::v3::ReadValue>();
    const auto in = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const string variable_id = "v0";
    const auto read_value = make_shared<ov::op::v3::ReadValue>(in, variable_id);
    NodeBuilder builder(read_value, {in});
    EXPECT_NO_THROW(auto g_read_value = ov::as_type_ptr<ov::op::v3::ReadValue>(builder.create()));

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, readvalue_v6_op) {
    NodeBuilder::get_ops().register_factory<ov::op::v6::ReadValue>();
    const auto in = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    const auto read_value = make_shared<ov::op::v6::ReadValue>(in, variable);
    NodeBuilder builder(read_value, {in});
    EXPECT_NO_THROW(auto g_read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(builder.create()));

    // attribute count
    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
