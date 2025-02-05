// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

namespace ov::test {
using op::v0::Parameter, op::v0::Constant;

TEST(attributes, istft) {
    NodeBuilder::opset().insert<ov::op::v16::ISTFT>();
    const auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{9, 9, 3});
    const auto window = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{16});
    const auto frame_size = Constant::create<int32_t>(ov::element::i32, {}, {16});
    const auto step_size = Constant::create<int32_t>(ov::element::i32, {}, {4});

    constexpr bool center = true;
    constexpr bool normalized = true;
    const auto op = std::make_shared<ov::op::v16::ISTFT>(data, window, frame_size, step_size, center, normalized);

    NodeBuilder builder(op, {data, window, frame_size, step_size});
    auto g_op = ov::as_type_ptr<ov::op::v16::ISTFT>(builder.create());

    constexpr auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_center(), g_op->get_center());
    EXPECT_EQ(op->get_normalized(), g_op->get_normalized());
}

TEST(attributes, istft_with_length) {
    NodeBuilder::opset().insert<ov::op::v16::ISTFT>();
    const auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{9, 9, 3});
    const auto window = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{16});
    const auto frame_size = Constant::create<int32_t>(ov::element::i32, {}, {16});
    const auto step_size = Constant::create<int32_t>(ov::element::i32, {}, {4});
    const auto signal_length = Constant::create<int32_t>(ov::element::i32, {}, {42});

    constexpr bool center = true;
    constexpr bool normalized = true;
    const auto op =
        std::make_shared<ov::op::v16::ISTFT>(data, window, frame_size, step_size, signal_length, center, normalized);

    NodeBuilder builder(op, {data, window, frame_size, step_size});
    auto g_op = ov::as_type_ptr<ov::op::v16::ISTFT>(builder.create());

    constexpr auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_center(), g_op->get_center());
    EXPECT_EQ(op->get_normalized(), g_op->get_normalized());
}
}  // namespace ov::test
