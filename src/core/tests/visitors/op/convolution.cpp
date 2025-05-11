// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convolution) {
    NodeBuilder::opset().insert<op::v1::Convolution>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 16, 124, 124});
    auto filters = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 16, 3, 3});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto convolution =
        make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, op::PadType::VALID);

    NodeBuilder builder(convolution, {data, filters});
    auto g_convolution = ov::as_type_ptr<op::v1::Convolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
}

TEST(attributes, convolution2) {
    NodeBuilder::opset().insert<op::v1::Convolution>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 227, 227});
    auto filters = make_shared<ov::op::v0::Parameter>(element::f32, Shape{96, 3, 227, 227});
    auto strides = Strides{4, 4};
    auto pads_begin = CoordinateDiff{0, 0};
    auto pads_end = CoordinateDiff{0, 0};
    auto dilations = Strides{1, 1};
    auto convolution =
        make_shared<op::v1::Convolution>(data, filters, strides, pads_begin, pads_end, dilations, op::PadType::VALID);
    NodeBuilder builder(convolution, {data, filters});
    auto g_convolution = ov::as_type_ptr<op::v1::Convolution>(builder.create());

    // attribute count
    const auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
}
