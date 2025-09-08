// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/clamp.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, clamp_op) {
    NodeBuilder::opset().insert<ov::op::v0::Clamp>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});

    double min = 0.4;
    double max = 5.6;

    const auto clamp = make_shared<op::v0::Clamp>(data, min, max);
    NodeBuilder builder(clamp, {data});
    auto g_clamp = ov::as_type_ptr<op::v0::Clamp>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_clamp->get_min(), clamp->get_min());
    EXPECT_EQ(g_clamp->get_max(), clamp->get_max());
}
