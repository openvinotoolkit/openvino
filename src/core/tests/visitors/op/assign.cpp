// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"
#include "visitors/visitors.hpp"

TEST(attributes, assign_v3_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v3::Assign>();
    const auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    const std::string variable_id = "v0";
    const auto read_value = std::make_shared<ov::op::v3::ReadValue>(in, variable_id);
    const auto assign = std::make_shared<ov::op::v3::Assign>(read_value, variable_id);
    ov::test::NodeBuilder builder(assign, {read_value});

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, assign_v6_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v6::Assign>();
    const auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    const auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
    const auto read_value = std::make_shared<ov::op::v6::ReadValue>(in, variable);
    const auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    ov::test::NodeBuilder builder(assign, {read_value});

    // attribute count
    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
