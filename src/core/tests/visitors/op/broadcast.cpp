// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, broadcast_v1) {
    NodeBuilder::opset().insert<ov::op::v1::Broadcast>();
    const auto arg = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = ov::op::AutoBroadcastType::NONE;

    const auto broadcast_v3 = make_shared<op::v1::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape, axes_mapping});
    auto g_broadcast_v3 = ov::as_type_ptr<ov::op::v1::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec().m_type, broadcast_spec);
}

TEST(attributes, broadcast_v3) {
    NodeBuilder::opset().insert<ov::op::v3::Broadcast>();
    const auto arg = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape});
    auto g_broadcast_v3 = ov::as_type_ptr<ov::op::v3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}

TEST(attributes, broadcast_v3_explicit) {
    NodeBuilder::opset().insert<ov::op::v3::Broadcast>();
    const auto arg = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    const auto axes_mapping = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::EXPLICIT;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape, axes_mapping});
    auto g_broadcast_v3 = ov::as_type_ptr<ov::op::v3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}
