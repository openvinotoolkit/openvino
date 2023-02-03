// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, broadcast_v1) {
    NodeBuilder::get_ops().register_factory<opset1::Broadcast>();
    const auto arg = make_shared<op::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = ov::op::AutoBroadcastType::NONE;

    const auto broadcast_v3 = make_shared<op::v1::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape, axes_mapping});
    auto g_broadcast_v3 = ov::as_type_ptr<opset1::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec().m_type, broadcast_spec);
}

TEST(attributes, broadcast_v3) {
    NodeBuilder::get_ops().register_factory<opset3::Broadcast>();
    const auto arg = make_shared<op::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape});
    auto g_broadcast_v3 = ov::as_type_ptr<opset3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}

TEST(attributes, broadcast_v3_explicit) {
    NodeBuilder::get_ops().register_factory<opset3::Broadcast>();
    const auto arg = make_shared<op::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto axes_mapping = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::EXPLICIT;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, axes_mapping, broadcast_spec);
    NodeBuilder builder(broadcast_v3, {arg, shape, axes_mapping});
    auto g_broadcast_v3 = ov::as_type_ptr<opset3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}
