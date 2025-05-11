// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, mvn_v1_op) {
    NodeBuilder::opset().insert<ov::op::v0::MVN>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4, 5});

    const auto axes = AxisSet{0, 1};

    const auto op = make_shared<ov::op::v0::MVN>(data, true, false, 0.1);
    op->set_reduction_axes(axes);
    NodeBuilder builder(op, {data});
    const auto g_op = ov::as_type_ptr<ov::op::v0::MVN>(builder.create());
    const auto expected_attr_count = 4;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_reduction_axes(), op->get_reduction_axes());
    EXPECT_EQ(g_op->get_across_channels(), op->get_across_channels());
    EXPECT_EQ(g_op->get_normalize_variance(), op->get_normalize_variance());
    EXPECT_EQ(g_op->get_eps(), op->get_eps());
}

TEST(attributes, mvn_v6_op) {
    NodeBuilder::opset().insert<ov::op::v6::MVN>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});

    const auto op = make_shared<ov::op::v6::MVN>(data, axes, false, 0.1f, op::MVNEpsMode::INSIDE_SQRT);

    NodeBuilder builder(op, {data, axes});
    const auto g_op = ov::as_type_ptr<ov::op::v6::MVN>(builder.create());
    const auto expected_attr_count = 3;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_eps_mode(), op->get_eps_mode());
    EXPECT_EQ(g_op->get_normalize_variance(), op->get_normalize_variance());
    EXPECT_EQ(g_op->get_eps(), op->get_eps());
}
