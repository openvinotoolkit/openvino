// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, grid_sample_defaults) {
    NodeBuilder::opset().insert<ov::op::v9::GridSample>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
    const auto grid = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 5, 5, 2});

    const auto op = make_shared<ov::op::v9::GridSample>(data, grid, ov::op::v9::GridSample::Attributes{});
    NodeBuilder builder(op, {data, grid});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v9::GridSample>(builder.create()));

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
