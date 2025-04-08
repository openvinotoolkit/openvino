// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include <gtest/gtest.h>

#include "openvino/op/unique.hpp"
#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, scaled_dot_product_attention) {
    NodeBuilder::opset().insert<ov::op::v13::ScaledDotProductAttention>();
    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 32, 32});
    const auto casual = false;

    const auto op = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, casual);
    NodeBuilder builder(op, {query, key, value});
    auto g_sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(builder.create());

    constexpr auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_causal(), g_sdpa->get_causal());
}
