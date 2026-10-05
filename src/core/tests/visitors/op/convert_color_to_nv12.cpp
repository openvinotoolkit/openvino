// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

namespace ov::tests {

TEST(attributes, convert_color_rgb_to_nv12_single_plane) {
    NodeBuilder::opset().insert<ov::op::v17::RGBtoNV12>();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3, 480, 640, 3});
    auto convert_color = std::make_shared<ov::op::v17::RGBtoNV12>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v17::RGBtoNV12>(builder.create()));

    const auto expected_attr_count = 1;  // single_plane attribute
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_rgb_to_nv12_two_planes) {
    NodeBuilder::opset().insert<ov::op::v17::RGBtoNV12>();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3, 480, 640, 3});
    auto convert_color = std::make_shared<ov::op::v17::RGBtoNV12>(data, false);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v17::RGBtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_bgr_to_nv12_single_plane) {
    NodeBuilder::opset().insert<ov::op::v17::BGRtoNV12>();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3, 480, 640, 3});
    auto convert_color = std::make_shared<ov::op::v17::BGRtoNV12>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v17::BGRtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_bgr_to_nv12_two_planes) {
    NodeBuilder::opset().insert<ov::op::v17::BGRtoNV12>();
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{3, 480, 640, 3});
    auto convert_color = std::make_shared<ov::op::v17::BGRtoNV12>(data, false);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v17::BGRtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
}  // namespace ov::tests
