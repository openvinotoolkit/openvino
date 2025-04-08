// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"

#include <gtest/gtest.h>

#include "openvino/op/unique.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, group_normalization) {
    NodeBuilder::opset().insert<ov::op::v12::GroupNormalization>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto scale = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    const auto bias = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});

    const auto op = make_shared<ov::op::v12::GroupNormalization>(data, scale, bias, 3, 0.00001f);
    NodeBuilder builder(op);
    auto g_unique = ov::as_type_ptr<ov::op::v12::GroupNormalization>(builder.create());

    constexpr auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(op->get_num_groups(), g_unique->get_num_groups());
    EXPECT_NEAR(op->get_epsilon(), g_unique->get_epsilon(), 0.00001f);
}
