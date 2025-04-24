// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convert_color_nv12_rgb) {
    NodeBuilder::opset().insert<op::v8::NV12toRGB>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 720, 640, 1});
    auto convert_color = make_shared<op::v8::NV12toRGB>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::NV12toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_nv12_bgr) {
    NodeBuilder::opset().insert<op::v8::NV12toBGR>();
    auto data = make_shared<op::v0::Parameter>(element::u8, Shape{3, 720, 640, 1});
    auto convert_color = make_shared<op::v8::NV12toBGR>(data);
    NodeBuilder builder(convert_color, {data});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::NV12toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_nv12_rgb_2planes) {
    NodeBuilder::opset().insert<op::v8::NV12toRGB>();
    auto data1 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 1});
    auto data2 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 2});
    auto convert_color = make_shared<op::v8::NV12toRGB>(data1, data2);
    NodeBuilder builder(convert_color, {data1, data2});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::NV12toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, convert_color_nv12_bgr_2planes) {
    NodeBuilder::opset().insert<op::v8::NV12toBGR>();
    auto data1 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 480, 640, 1});
    auto data2 = make_shared<op::v0::Parameter>(element::u8, Shape{3, 240, 320, 2});
    auto convert_color = make_shared<op::v8::NV12toBGR>(data1, data2);
    NodeBuilder builder(convert_color, {data1, data2});
    EXPECT_NO_THROW(auto g_concat = ov::as_type_ptr<op::v8::NV12toRGB>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
