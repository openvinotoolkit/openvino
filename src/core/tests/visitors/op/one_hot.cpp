// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, one_hot_op) {
    NodeBuilder::opset().insert<ov::op::v1::OneHot>();
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = ov::op::v0::Constant::create(element::i64, Shape{}, {4});
    auto on_value = ov::op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = ov::op::v0::Constant::create(element::f32, Shape{}, {0.0f});

    int64_t axis = 3;

    auto one_hot = make_shared<ov::op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    NodeBuilder builder(one_hot, {indices, depth, on_value, off_value});
    auto g_one_hot = ov::as_type_ptr<ov::op::v1::OneHot>(builder.create());

    EXPECT_EQ(g_one_hot->get_axis(), one_hot->get_axis());
}
