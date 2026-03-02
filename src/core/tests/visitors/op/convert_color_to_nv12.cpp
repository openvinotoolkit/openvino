// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/rgb_to_nv12.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convert_color_rgb_to_nv12_single_plane) {
    NodeBuilder::opset().insert<op::v16::RGBtoNV12>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 3});
    auto convert_color = make_shared<op::v16::RGBtoNV12>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v16::RGBtoNV12>(builder.create()));

    const auto expected_attr_count = 1;  // single_plane attribute
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_rgb_to_nv12_two_planes) {
    NodeBuilder::opset().insert<op::v16::RGBtoNV12>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 3});
    auto convert_color = make_shared<op::v16::RGBtoNV12>(data, false);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v16::RGBtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_bgr_to_nv12_single_plane) {
    NodeBuilder::opset().insert<op::v16::BGRtoNV12>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 3});
    auto convert_color = make_shared<op::v16::BGRtoNV12>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v16::BGRtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_bgr_to_nv12_two_planes) {
    NodeBuilder::opset().insert<op::v16::BGRtoNV12>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 3});
    auto convert_color = make_shared<op::v16::BGRtoNV12>(data, false);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v16::BGRtoNV12>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
