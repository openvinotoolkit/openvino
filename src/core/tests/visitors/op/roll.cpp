// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, roll_op) {
    NodeBuilder::opset().insert<ov::op::v7::Roll>();
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 3});
    const auto B = make_shared<ov::op::v0::Constant>(element::i32, Shape{3});
    const auto C = make_shared<ov::op::v0::Constant>(element::i32, Shape{3});

    const auto roll = make_shared<ov::op::v7::Roll>(A, B, C);
    NodeBuilder builder(roll, {A, B, C});
    EXPECT_NO_THROW(auto g_roll = ov::as_type_ptr<ov::op::v7::Roll>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
