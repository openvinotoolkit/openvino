// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/convolution.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convolution_backprop_op) {
    NodeBuilder::opset().insert<ov::op::v1::ConvolutionBackpropData>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 16, 124, 124});
    auto filters = make_shared<ov::op::v0::Parameter>(element::f32, Shape{16, 2, 3, 3});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto convolution = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                        filters,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        op::PadType::VALID);
    NodeBuilder builder(convolution, {data, filters});
    auto g_convolution = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(builder.create());

    // attribute count
    const auto expected_attr_count = 6;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
    EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
    EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
    EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
    EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
}

TEST(attributes, convolution_backprop_output_shape_output_padding) {
    NodeBuilder::opset().insert<ov::op::v1::ConvolutionBackpropData>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 16, 124, 124});
    const auto filter = make_shared<ov::op::v0::Parameter>(element::f32, Shape{16, 2, 3, 3});
    const auto output_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    const auto strides = Strides{2, 1};
    const auto pads_begin = CoordinateDiff{3, 4};
    const auto pads_end = CoordinateDiff{4, 6};
    const auto dilations = Strides{3, 1};
    const auto output_padding = CoordinateDiff{3, 4};

    const std::initializer_list<op::PadType> allPadTypes = {op::PadType::EXPLICIT,
                                                            op::PadType::SAME_UPPER,
                                                            op::PadType::SAME_LOWER,
                                                            op::PadType::VALID,
                                                            op::PadType::AUTO,
                                                            op::PadType::NOTSET};

    for (auto padType : allPadTypes) {
        const auto convolution = make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                                  filter,
                                                                                  output_shape,
                                                                                  strides,
                                                                                  pads_begin,
                                                                                  pads_end,
                                                                                  dilations,
                                                                                  padType,
                                                                                  output_padding);
        NodeBuilder builder(convolution, {data, filter});
        const auto g_convolution = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(builder.create());

        // attribute count
        const auto expected_attr_count = 6;
        EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

        EXPECT_EQ(g_convolution->get_strides(), convolution->get_strides());
        EXPECT_EQ(g_convolution->get_pads_begin(), convolution->get_pads_begin());
        EXPECT_EQ(g_convolution->get_pads_end(), convolution->get_pads_end());
        EXPECT_EQ(g_convolution->get_dilations(), convolution->get_dilations());
        EXPECT_EQ(g_convolution->get_auto_pad(), convolution->get_auto_pad());
        EXPECT_EQ(g_convolution->get_output_padding(), convolution->get_output_padding());
    }
}
