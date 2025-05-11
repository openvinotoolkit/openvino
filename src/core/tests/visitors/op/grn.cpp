// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grn.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, grn_op) {
    NodeBuilder::opset().insert<ov::op::v0::GRN>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});

    float bias = 1.25f;

    auto grn = make_shared<ov::op::v0::GRN>(data, bias);
    NodeBuilder builder(grn, {data});
    auto g_grn = ov::as_type_ptr<ov::op::v0::GRN>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_grn->get_bias(), grn->get_bias());
}
