// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/broadcast.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, broadcast_v3) {
    NodeBuilder::get_ops().register_factory<op::v3::Broadcast>();
    const auto arg = make_shared<op::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    NodeBuilder builder(broadcast_v3);
    auto g_broadcast_v3 = as_type_ptr<op::v3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}
