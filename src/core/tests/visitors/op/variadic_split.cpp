// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, variadic_split_op) {
    NodeBuilder::opset().insert<op::v1::VariadicSplit>();
    auto data = make_shared<op::v0::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::v0::Parameter>(element::i32, Shape{1});
    auto split_lengths = make_shared<op::v0::Parameter>(element::i32, Shape{1});

    auto split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);
    NodeBuilder builder(split, {data, axis, split_lengths});
    EXPECT_NO_THROW(auto g_split = ov::as_type_ptr<op::v1::VariadicSplit>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
