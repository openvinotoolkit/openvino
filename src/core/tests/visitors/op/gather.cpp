// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, gather_v1_op) {
    NodeBuilder::opset().insert<ov::op::v1::Gather>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 2);

    auto gather = make_shared<ov::op::v1::Gather>(data, indices, axis);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<ov::op::v1::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v7_op) {
    NodeBuilder::opset().insert<ov::op::v7::Gather>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<ov::op::v7::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<ov::op::v7::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v8_op) {
    NodeBuilder::opset().insert<ov::op::v8::Gather>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<ov::op::v8::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather, {data, indices, axis});
    auto g_gather = ov::as_type_ptr<ov::op::v8::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}
