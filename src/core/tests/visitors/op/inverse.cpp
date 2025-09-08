// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include <gtest/gtest.h>

#include "openvino/op/unique.hpp"
#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, inverse) {
    NodeBuilder::opset().insert<ov::op::v14::Inverse>();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});

    const auto op = std::make_shared<ov::op::v14::Inverse>(data, true);
    NodeBuilder builder(op, {data});
    auto g_inv = ov::as_type_ptr<ov::op::v14::Inverse>(builder.create());

    constexpr auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_adjoint(), g_inv->get_adjoint());
}
