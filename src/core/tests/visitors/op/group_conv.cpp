// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, group_conv_op) {
    NodeBuilder::get_ops().register_factory<opset1::GroupConvolution>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 12, 224, 224});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 1, 3, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto group_conv = make_shared<opset1::GroupConvolution>(data,
                                                            filters,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            dilations,
                                                            op::PadType::VALID);
    NodeBuilder builder(group_conv);
    auto g_group_conv = ov::as_type_ptr<opset1::GroupConvolution>(builder.create());
    EXPECT_EQ(g_group_conv->get_strides(), group_conv->get_strides());
    EXPECT_EQ(g_group_conv->get_pads_begin(), group_conv->get_pads_begin());
    EXPECT_EQ(g_group_conv->get_pads_end(), group_conv->get_pads_end());
    EXPECT_EQ(g_group_conv->get_dilations(), group_conv->get_dilations());
    EXPECT_EQ(g_group_conv->get_auto_pad(), group_conv->get_auto_pad());
}

TEST(attributes, group_conv_backprop_data_op) {
    NodeBuilder::get_ops().register_factory<opset1::GroupConvolutionBackpropData>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 20, 224, 224});
    const auto filter = make_shared<op::Parameter>(element::f32, Shape{4, 5, 2, 3, 3});
    const auto output_shape = make_shared<op::Parameter>(element::i32, Shape{2});

    const auto strides = Strides{2, 1};
    const auto pads_begin = CoordinateDiff{3, 4};
    const auto pads_end = CoordinateDiff{4, 6};
    const auto dilations = Strides{3, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const auto output_padding = CoordinateDiff{3, 4};

    const auto gcbd = make_shared<opset1::GroupConvolutionBackpropData>(data,
                                                                        filter,
                                                                        output_shape,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad,
                                                                        output_padding);
    NodeBuilder builder(gcbd);
    const auto g_gcbd = ov::as_type_ptr<opset1::GroupConvolutionBackpropData>(builder.create());

    EXPECT_EQ(g_gcbd->get_strides(), gcbd->get_strides());
    EXPECT_EQ(g_gcbd->get_pads_begin(), gcbd->get_pads_begin());
    EXPECT_EQ(g_gcbd->get_pads_end(), gcbd->get_pads_end());
    EXPECT_EQ(g_gcbd->get_dilations(), gcbd->get_dilations());
    EXPECT_EQ(g_gcbd->get_auto_pad(), gcbd->get_auto_pad());
    EXPECT_EQ(g_gcbd->get_output_padding(), gcbd->get_output_padding());
}
