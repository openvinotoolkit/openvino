//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "op/convolution.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// ---------------------------- v0 ----------------------------
TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 91}); // output delta
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             Strides{1},
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96}); // output delta
    auto move_strides = Strides{1};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_strided)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 46}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_strided_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 48}); // output delta
    auto move_strides = Strides{2};
    auto dilation_strides = Strides{1};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_strided_small_uneven)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 5};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 2}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_strided_small_even)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 6};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 2});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 3}); // output delta
    auto move_strides = Strides{2};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             Strides{1},
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_window_dilated)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 82}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             dilate_strides,
                                                             CoordinateDiff{0},
                                                             CoordinateDiff{0},
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{0});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{0});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_window_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});  // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 87}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             dilate_strides,
                                                             padding_below,
                                                             padding_above,
                                                             Strides{1});
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{1});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

TEST(type_prop, conv_backprop_data_v0_1d_batch_deduce_window_dilated_data_dilated_padded)
{
    // Deduce type
    Shape data_batch_shape{64, 3, 100};
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});   // filters
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{64, 128, 285}); // output delta
    auto move_strides = Strides{1};
    auto dilate_strides = Strides{2};
    auto padding_below = CoordinateDiff{2};
    auto padding_above = CoordinateDiff{3};
    auto data_dilate_strides = Strides{3};
    auto conv = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                             param0,
                                                             param1,
                                                             move_strides,
                                                             dilate_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilate_strides);
    EXPECT_EQ(conv->get_element_type(), element::f32);
    EXPECT_EQ(conv->get_shape(), data_batch_shape);

    EXPECT_EQ(conv->get_window_movement_strides_forward(), Strides{1});
    EXPECT_EQ(conv->get_window_dilation_strides_forward(), Strides{2});
    EXPECT_EQ(conv->get_data_dilation_strides_forward(), Strides{3});

    EXPECT_EQ(conv->get_padding_below_forward(), CoordinateDiff{2});
    EXPECT_EQ(conv->get_padding_above_forward(), CoordinateDiff{3});
}

// ---------------------------- v1 ----------------------------
TEST(type_prop, convolution_backprop_data_partial_auto_padding_upper)
{
    const Shape shape1{1, 512, 1, 37};
    const Shape shape2{512, 256, 1, 1};
    const Shape shape3{2};
    Strides strides{1, 2};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto in1 = make_shared<op::Parameter>(element::f32, shape1);
    auto in2 = make_shared<op::Parameter>(element::f32, shape2);
    std::vector<int64_t> data = {1, 74};
    element::Type type = element::i64;
    auto in3 = make_shared<op::Constant>(type, shape3, data);

    auto conv = make_shared<op::v1::ConvolutionBackpropData>(
        in1, in2, in3, strides, pads_begin, pads_end, dilations, auto_pad);
    conv->validate_and_infer_types();

    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_backprop_data_partial_auto_padding_lower)
{
    const Shape shape1{1, 512, 1, 37};
    const Shape shape2{512, 256, 1, 1};
    const Shape shape3{2};
    Strides strides{1, 2};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto in1 = make_shared<op::Parameter>(element::f32, shape1);
    auto in2 = make_shared<op::Parameter>(element::f32, shape2);
    std::vector<int64_t> data = {1, 74};
    element::Type type = element::i64;
    auto in3 = make_shared<op::Constant>(type, shape3, data);

    auto conv = make_shared<op::v1::ConvolutionBackpropData>(
        in1, in2, in3, strides, pads_begin, pads_end, dilations, auto_pad);
    conv->validate_and_infer_types();

    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{0, 0}));
}

TEST(type_prop, convolution_backprop_data_auto_pad_explicit_with_output_padding)
{
    const PartialShape data_pshape{1, 16, 2, 2};
    const PartialShape filters_pshape{16, 6, 3, 3};

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};
    const CoordinateDiff output_padding{1, 1};
    const op::PadType auto_pad = op::PadType::EXPLICIT;

    const element::Type_t inputs_et = element::f16;
    auto data = make_shared<op::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<op::Parameter>(inputs_et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations, auto_pad, output_padding);
    
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(PartialShape{1, 6, 4, 4}));
    ASSERT_EQ(conv_backprop->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv_backprop->get_pads_end(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv_backprop->get_output_padding(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_backprop_data_auto_pad_same_with_output_padding_and_output_shape)
{
    const PartialShape data_pshape{1, 16, 2, 2};
    const PartialShape filters_pshape{16, 6, 3, 3};

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};
    const CoordinateDiff output_padding{1, 1};
    const op::PadType auto_pad = op::PadType::SAME_LOWER;

    const element::Type_t inputs_et = element::f16;
    auto data = make_shared<op::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<op::Parameter>(inputs_et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, output_shape, strides, padding_begin, padding_end, dilations, auto_pad, output_padding);
    
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(PartialShape{1, 6, 3, 3}));
    ASSERT_EQ(conv_backprop->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv_backprop->get_pads_end(), (CoordinateDiff{2, 2}));
    ASSERT_EQ(conv_backprop->get_output_padding(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, convolution_backprop_data_output_shape_as_const)
{
    const PartialShape data_pshape{1, 16, 5, 5};
    const PartialShape filters_pshape{16, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);

    EXPECT_EQ(conv_backprop->get_element_type(), element::f32);
    EXPECT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 3, 3}));
    EXPECT_EQ(conv_backprop->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv_backprop->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(conv_backprop->get_pads_begin(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(conv_backprop->get_pads_end(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(conv_backprop->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv_backprop->get_auto_pad(), op::PadType::SAME_UPPER);
}

TEST(type_prop, convolution_backprop_data_output_shape_as_param)
{
    const PartialShape data_pshape{1, 16, 5, 5};
    const PartialShape filters_pshape{16, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);

    EXPECT_EQ(conv_backprop->get_element_type(), element::f32);
    EXPECT_EQ(conv_backprop->get_auto_pad(), op::PadType::SAME_UPPER);
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_data_nc_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{16, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_filters_cin_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 16, 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), 6, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 6, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_static_ranks_filters_cin_cout_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 16, 5, 5};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                      filters,
                                                                      output_shape,
                                                                      Strides{},
                                                                      CoordinateDiff{},
                                                                      CoordinateDiff{},
                                                                      Strides{},
                                                                      op::PadType::SAME_UPPER);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_data_nc_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), Dimension::dynamic(), 224, 224};
    const PartialShape filters_pshape{5, 2, 3, 3};
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_cin_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 20, 224, 224};
    const PartialShape filters_pshape{Dimension::dynamic(), 2, 3, 3};
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_cin_cout_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 20, 224, 224};
    const PartialShape filters_pshape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_data_spatial_dims_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 4, Dimension::dynamic(), 224};
    const PartialShape filters_pshape{4, 16, 3, 3};
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 16, Dimension::dynamic(), 447}));
}

TEST(type_prop, convolution_backprop_data_dyn_static_ranks_filters_spatial_dims_dyn)
{
    const PartialShape data_pshape{Dimension::dynamic(), 4, 224, 224};
    const PartialShape filters_pshape{4, 16, 3, Dimension::dynamic()};
    const element::Type_t et = element::f32;

    const Strides strides{2, 2};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 16, 447, Dimension::dynamic()}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_data_batch)
{
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{16, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_dyn_filters)
{
    const PartialShape data_pshape{1, 16, Dimension::dynamic(), Dimension::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_as_const_dyn_data_and_filters)
{
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = op::Constant::create(element::i64, Shape{3}, {3, 3, 3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{5}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3, 3}));
}

TEST(type_prop, convolution_backprop_data_with_output_shape_as_param_dyn_data_and_filters)
{
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto output_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, convolution_backprop_data_shape_dyn_data)
{
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{4, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, convolution_backprop_data_shape_dyn_filters)
{
    const PartialShape data_pshape{1, 4, 224, 224}; // [N, C_IN, H, W]
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, convolution_backprop_data_dyn_data_and_filters)
{
    const PartialShape data_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data = make_shared<op::Parameter>(et, data_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv_backprop->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, convolution_backprop_data_invalid_et_inputs)
{
    const PartialShape data_pshape{1, 16, 5, 5};
    const PartialShape filters_pshape{16, 6, 3, 3};

    const Strides strides{1, 1};
    const Strides dilations{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};

    try
    {
        const element::Type_t data_et = element::f16;
        const element::Type_t filters_et = element::i64;

        auto data = make_shared<op::Parameter>(data_et, data_pshape);
        auto filters = make_shared<op::Parameter>(filters_et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, padding_begin, padding_end, dilations);
        FAIL() << "Invalid element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element types for data batch and filters do not match");
    }
    catch (...)
    {
        FAIL() << "Element types of data batch and filters validation check failed for unexpected "
                  "reason.";
    }

    try
    {
        const element::Type_t input_et = element::boolean;

        auto data = make_shared<op::Parameter>(input_et, data_pshape);
        auto filters = make_shared<op::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, padding_begin, padding_end, dilations);
        FAIL() << "Invalid element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be numeric");
    }
    catch (...)
    {
        FAIL() << "Numeric element types of data batch and filters validation check failed for "
                  "unexpected reason.";
    }

    try
    {
        const element::Type_t et = element::f32;

        auto data = make_shared<op::Parameter>(et, data_pshape);
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto output_shape = op::Constant::create(et, Shape{2}, {3, 3});
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(data,
                                                                          filters,
                                                                          output_shape,
                                                                          Strides{},
                                                                          CoordinateDiff{},
                                                                          CoordinateDiff{},
                                                                          Strides{},
                                                                          op::PadType::SAME_UPPER);
        // output shape input element type must be of integer type
        FAIL() << "Invalid element type of output_shape input not detected";
    }
    catch(const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type for output shape should be of integer type");
    }
    catch (...)
    {
        FAIL() << "Element type of output_shape input validation check failed for unexpected reason";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_input_ranks)
{
    const element::Type_t input_et = element::f32;

    // data and filters don't have same rank
    try
    {
        const PartialShape data_pshape{1, 20, 224, 224, 224};
        const PartialShape filters_pshape{20, 10, 3, 3};

        auto data = make_shared<op::Parameter>(input_et, data_pshape);
        auto filters = make_shared<op::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data and filters inputs must have same rank");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data and filters don't have spatial dimensions
    try
    {
        const PartialShape data_pshape{1, 20};
        const PartialShape filters_pshape{20, 10};

        auto data = make_shared<op::Parameter>(input_et, data_pshape);
        auto filters = make_shared<op::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data and filters inputs must have rank 3, 4 or 5");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data and filters have 4 spatial dimensions (not supported)
    try
    {
        const PartialShape data_pshape{1, 20, 224, 224, 224, 224};
        const PartialShape filters_pshape{20, 10, 3, 3, 3, 3};

        auto data = make_shared<op::Parameter>(input_et, data_pshape);
        auto filters = make_shared<op::Parameter>(input_et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Data and filters inputs must have rank 3, 4 or 5");
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    try
    {
        const PartialShape data_pshape{1, 16, 5, 5};
        const PartialShape filters_shape{16, 2, 3, 3};

        auto data = make_shared<op::Parameter>(input_et, data_pshape);
        auto filters = make_shared<op::Parameter>(input_et, filters_shape);
        auto output_shape = op::Constant::create(element::i64, Shape{3, 1}, {3, 3, 3});
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // output_shape has rank 2, should be rank 1
        FAIL() << "Incompatible rank of output shape optional input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Spatial shape of output input must be of rank 1"));
    }
    catch (...)
    {
        FAIL() << "Output shape rank validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_input_channel_dims)
{
    const PartialShape data_pshape{1, 32, 5, 5};
    const PartialShape filters_pshape{16, 20, 3, 3};
    const element::Type_t inputs_et = element::f32;

    Strides strides{1, 1};
    Strides dilations{1, 1};
    CoordinateDiff padding{2, 2};

    auto data = make_shared<op::Parameter>(inputs_et, data_pshape);
    auto filters = make_shared<op::Parameter>(inputs_et, filters_pshape);
    try
    {
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, padding, padding, dilations);
        // data input shape does not have correct dimension C_IN
        FAIL() << "Incompatibile input shapes not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input channels dimension of data and filters inputs must be equal"));
    }
    catch (...)
    {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_output_shape_spatial_dims)
{
    const PartialShape data_pshape{1, 16, 5, 5};
    const PartialShape filters_shape{16, 2, 3, 3};
    const element::Type_t inputs_et = element::f32;

    try
    {
        auto data = make_shared<op::Parameter>(inputs_et, data_pshape);
        auto filters = make_shared<op::Parameter>(inputs_et, filters_shape);
        auto output_shape = op::Constant::create(element::i64, Shape{3}, {3, 3, 3});
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, output_shape, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // output_shape has invalid spatials dimensions (should be 2)
        FAIL() << "Incompatible output shape optional input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Output shape should be specified only and for all spatial dimensions."));
    }
    catch (...)
    {
        FAIL() << "Output shape validation check failed for unexpected reason.";
    }
}

TEST(type_prop, convolution_backprop_data_invalid_conv_param_spatial_dims)
{
    const PartialShape data_pshape{1, 20, 224, 224};
    const PartialShape filters_pshape{20, 10, 3, 3};
    const element::Type_t et = element::f32;

    // invalid strides spatial dimensions
    try
    {
        Strides strides{1, 1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data = make_shared<op::Parameter>(et, data_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Strides should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid strides spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Strides should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Strides spatial dimensions validation check failed for unexpected reason";
    }

    // invalid dilations spatial dimensions
    try
    {
        Strides strides{1, 1};
        Strides dilations{1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data = make_shared<op::Parameter>(et, data_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Dilations should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid dilations spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Dilations should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Dilations spatial dimensions validation check failed for unexpected reason";
    }

    // invalid padding spatial dimensions
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data = make_shared<op::Parameter>(et, data_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Pads should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0};

        auto data = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations);
        FAIL() << "Invalid padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Pads should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Padding spatial dimensions validation check failed for unexpected reason";
    }

    // invalid output padding spatial dimensions
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};
        CoordinateDiff output_padding{0, 0, 0};
        op::PadType auto_pad = op::PadType::EXPLICIT;

        auto data = make_shared<op::Parameter>(et, data_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations, auto_pad, output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output padding should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
    try
    {
        Strides strides{1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};
        CoordinateDiff output_padding{0};
        op::PadType auto_pad = op::PadType::EXPLICIT;

        auto data = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto conv_backprop = make_shared<op::v1::ConvolutionBackpropData>(
            data, filters, strides, pads_begin, pads_end, dilations, auto_pad, output_padding);
        FAIL() << "Invalid output padding spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Output padding should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Output padding spatial dimensions validation check failed for unexpected reason";
    }
}