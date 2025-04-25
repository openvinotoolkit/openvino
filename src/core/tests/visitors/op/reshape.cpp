// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, reshape_op) {
    NodeBuilder::opset().insert<ov::op::v1::Reshape>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto pattern = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    bool special_zero = true;

    auto reshape = make_shared<ov::op::v1::Reshape>(data, pattern, special_zero);
    NodeBuilder builder(reshape, {data, pattern});
    auto g_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(builder.create());

    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_reshape->get_special_zero(), reshape->get_special_zero());
}
