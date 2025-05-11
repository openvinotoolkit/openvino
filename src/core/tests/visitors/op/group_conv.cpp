// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_conv.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, group_conv_op) {
    NodeBuilder::opset().insert<ov::op::v1::GroupConvolution>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 12, 224, 224});
    auto filters = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 1, 3, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto group_conv = make_shared<op::v1::GroupConvolution>(data,
                                                            filters,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            dilations,
                                                            op::PadType::VALID);
    NodeBuilder builder(group_conv, {data, filters});
    auto g_group_conv = ov::as_type_ptr<op::v1::GroupConvolution>(builder.create());
    EXPECT_EQ(g_group_conv->get_strides(), group_conv->get_strides());
    EXPECT_EQ(g_group_conv->get_pads_begin(), group_conv->get_pads_begin());
    EXPECT_EQ(g_group_conv->get_pads_end(), group_conv->get_pads_end());
    EXPECT_EQ(g_group_conv->get_dilations(), group_conv->get_dilations());
    EXPECT_EQ(g_group_conv->get_auto_pad(), group_conv->get_auto_pad());
}

TEST(attributes, group_conv_backprop_data_op) {
    NodeBuilder::opset().insert<op::v1::GroupConvolutionBackpropData>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 20, 224, 224});
    const auto filter = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4, 5, 2, 3, 3});
    const auto output_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    const auto strides = Strides{2, 1};
    const auto pads_begin = CoordinateDiff{3, 4};
    const auto pads_end = CoordinateDiff{4, 6};
    const auto dilations = Strides{3, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const auto output_padding = CoordinateDiff{3, 4};

    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                        filter,
                                                                        output_shape,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad,
                                                                        output_padding);
    NodeBuilder builder(gcbd, {data, filter});
    const auto g_gcbd = ov::as_type_ptr<op::v1::GroupConvolutionBackpropData>(builder.create());

    EXPECT_EQ(g_gcbd->get_strides(), gcbd->get_strides());
    EXPECT_EQ(g_gcbd->get_pads_begin(), gcbd->get_pads_begin());
    EXPECT_EQ(g_gcbd->get_pads_end(), gcbd->get_pads_end());
    EXPECT_EQ(g_gcbd->get_dilations(), gcbd->get_dilations());
    EXPECT_EQ(g_gcbd->get_auto_pad(), gcbd->get_auto_pad());
    EXPECT_EQ(g_gcbd->get_output_padding(), gcbd->get_output_padding());
}
