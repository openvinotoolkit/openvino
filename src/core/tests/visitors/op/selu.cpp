// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selu.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, selu_op) {
    NodeBuilder::opset().insert<ov::op::v0::Selu>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto alpha = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto lambda = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    const auto op = make_shared<ov::op::v0::Selu>(data_input, alpha, lambda);

    NodeBuilder builder(op, {data_input, alpha, lambda});
    const auto expected_attr_count = 0;
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v0::Selu>(builder.create()));

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
