// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/divide.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, divide) {
    NodeBuilder::opset().insert<ov::op::v1::Divide>();

    const auto in1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    const auto in2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    const bool pythondiv = true;
    const op::AutoBroadcastSpec& auto_broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    const auto divide = make_shared<ov::op::v1::Divide>(in1, in2, pythondiv, auto_broadcast);

    NodeBuilder builder(divide, {in1, in2});
    auto g_divide = ov::as_type_ptr<ov::op::v1::Divide>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_divide->is_pythondiv(), divide->is_pythondiv());
    EXPECT_EQ(g_divide->get_autob(), divide->get_autob());
}
