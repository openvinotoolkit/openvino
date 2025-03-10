// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

namespace ov {
namespace test {
TEST(attributes, stft) {
    NodeBuilder::opset().insert<ov::op::v15::STFT>();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 48});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {16});
    const auto step_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {4});

    constexpr bool transpose_frames = true;
    const auto op = std::make_shared<ov::op::v15::STFT>(data, window, frame_size, step_size, transpose_frames);

    NodeBuilder builder(op, {data, window, frame_size, step_size});
    auto g_op = ov::as_type_ptr<ov::op::v15::STFT>(builder.create());

    constexpr auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_transpose_frames(), g_op->get_transpose_frames());
}
}  // namespace test
}  // namespace ov
