// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/Identity.hpp"

#include <gtest/gtest.h>

#include "openvino/op/unique.hpp"
#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, Identity) {
    NodeBuilder::opset().insert<ov::op::v15::Identity>();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2});

    const auto op = std::make_shared<ov::op::v15::Identity>(data);
    NodeBuilder builder(op, {data});
    auto g_identity = ov::as_type_ptr<ov::op::v15::Identity>(builder.create());

    constexpr auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
