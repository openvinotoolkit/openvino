// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/search_sorted.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, search_sorted_op) {
    using TOp = ov::op::v15::SearchSorted;
    NodeBuilder::opset().insert<TOp>();
    auto sorted = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 50, 50});
    auto values = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 50, 50});

    auto op = make_shared<TOp>(sorted, values);
    NodeBuilder builder(op, {sorted, values});
    auto g_op = ov::as_type_ptr<TOp>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // space_to_depth attributes
    EXPECT_EQ(g_op->get_right_mode(), op->get_right_mode());
}
