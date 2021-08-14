// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/gather.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_v7_op) {
    NodeBuilder::get_ops().register_factory<op::v7::Gather>();
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<op::v0::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<op::v7::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather);
    auto g_gather = as_type_ptr<op::v7::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v8_op) {
    NodeBuilder::get_ops().register_factory<op::v8::Gather>();
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<op::v0::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<op::v8::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather);
    auto g_gather = as_type_ptr<op::v8::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}
