// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, concat_op) {
    NodeBuilder::opset().insert<ov::op::v0::Concat>();
    auto input1 = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 2, 3});
    auto input2 = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 2, 3});
    auto input3 = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 2, 3});
    int64_t axis = 2;

    auto concat = make_shared<ov::op::v0::Concat>(ov::NodeVector{input1, input2, input3}, axis);
    NodeBuilder builder(concat, {input1, input2, input3});
    auto g_concat = ov::as_type_ptr<ov::op::v0::Concat>(builder.create());

    EXPECT_EQ(g_concat->get_axis(), concat->get_axis());
}
