// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/split.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, split_op) {
    NodeBuilder::opset().insert<ov::op::v1::Split>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto num_splits = 2;
    auto split = make_shared<ov::op::v1::Split>(data, axis, num_splits);
    NodeBuilder builder(split, {data, axis});
    auto g_split = ov::as_type_ptr<ov::op::v1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}

TEST(attributes, split_op2) {
    NodeBuilder::opset().insert<ov::op::v1::Split>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{300});
    auto axis = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto num_splits = 3;
    auto split = make_shared<ov::op::v1::Split>(data, axis, num_splits);
    NodeBuilder builder(split, {data, axis});
    auto g_split = ov::as_type_ptr<ov::op::v1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}
