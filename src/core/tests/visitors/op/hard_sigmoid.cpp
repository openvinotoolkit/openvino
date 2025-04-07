// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hard_sigmoid.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, hardsigmoid_op) {
    NodeBuilder::opset().insert<ov::op::v0::HardSigmoid>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 5});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    const auto beta = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    const auto hardsigmoid = make_shared<ov::op::v0::HardSigmoid>(data, alpha, beta);
    NodeBuilder builder(hardsigmoid, {data, alpha, beta});
    EXPECT_NO_THROW(auto g_hardsigmoid = ov::as_type_ptr<ov::op::v0::HardSigmoid>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
