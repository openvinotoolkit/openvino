// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, range_op) {
    NodeBuilder::opset().insert<ov::op::v4::Range>();
    auto start = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto stop = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto step = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto output_type = element::f32;

    auto range = make_shared<ov::op::v4::Range>(start, stop, step, output_type);
    NodeBuilder builder(range, {start, stop, step});
    auto g_range = ov::as_type_ptr<ov::op::v4::Range>(builder.create());

    EXPECT_EQ(g_range->get_output_type(), range->get_output_type());
}
