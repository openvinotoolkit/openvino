// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, reverse_op_enum_mode) {
    NodeBuilder::opset().insert<ov::op::v1::Reverse>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});

    auto reverse = make_shared<ov::op::v1::Reverse>(data, reversed_axes, ov::op::v1::Reverse::Mode::INDEX);
    NodeBuilder builder(reverse, {data, reversed_axes});
    auto g_reverse = ov::as_type_ptr<ov::op::v1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_op_string_mode) {
    NodeBuilder::opset().insert<ov::op::v1::Reverse>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});

    std::string mode = "index";

    auto reverse = make_shared<ov::op::v1::Reverse>(data, reversed_axes, mode);
    NodeBuilder builder(reverse, {data, reversed_axes});
    auto g_reverse = ov::as_type_ptr<ov::op::v1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}
