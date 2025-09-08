// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/idft.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, idft_op) {
    NodeBuilder::opset().insert<op::v7::IDFT>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2});
    auto idft = make_shared<op::v7::IDFT>(data, axes);

    NodeBuilder builder(idft, {data, axes});
    EXPECT_NO_THROW(auto g_idft = ov::as_type_ptr<op::v7::IDFT>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, idft_op_signal) {
    NodeBuilder::opset().insert<op::v7::IDFT>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 10, 10, 2});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {2});
    auto signal = ov::op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{1}, {20});
    auto idft = make_shared<op::v7::IDFT>(data, axes, signal);

    NodeBuilder builder(idft, {data, axes, signal});
    EXPECT_NO_THROW(auto g_idft = ov::as_type_ptr<op::v7::IDFT>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
