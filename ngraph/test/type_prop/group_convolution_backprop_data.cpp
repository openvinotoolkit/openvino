//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, group_conv_backprop_data)
{
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 8, 2, 3, 3});
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 6, 6});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
    EXPECT_EQ(gcbd->get_element_type(), element::f32);
    EXPECT_EQ(gcbd->get_output_shape(0), (Shape{1, 4, 8, 8}));
    EXPECT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_pads_end(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_backprop_data_output_shape_as_const)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    EXPECT_EQ(gcbd->get_element_type(), element::f32);
    EXPECT_EQ(gcbd->get_output_shape(0), (Shape{1, 2, 3, 3}));
    EXPECT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gcbd->get_pads_end(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::SAME_UPPER);
}

TEST(type_prop, group_conv_backprop_data_output_shape_as_param)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
    const auto output_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    EXPECT_EQ(gcbd->get_element_type(), element::f32);
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{1, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_static_ranks_shape_inference_1)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        gcbd->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_static_ranks_shape_inference_2)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 16, 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        gcbd->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_static_ranks_shape_inference_3)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 16, 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_static_ranks_shape_inference_4)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        gcbd->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_static_ranks_shape_inference_5)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 16, 5, 5});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 16, Dimension::dynamic(), 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_dyn_static_ranks_shape_inference_1)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 224, 224});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape{4, 5, 2, 3, 3});
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 8, 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_dyn_static_ranks_shape_inference_2)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 20, 224, 224});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 5, 2, 3, 3});
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(PartialShape{1, 8, 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_dyn_static_ranks_shape_inference_3)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 20, 224, 224});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3});
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_dyn_static_ranks_shape_inference_4)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 224, 224});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 5, 2, 3, 3});
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_dyn_static_ranks_shape_inference_5)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 20, 224, 224});
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 2, 3, 3});
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_data_batch)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        gcbd->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_shape_dyn_data_batch)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape{4, 5, 2, 3, 3});
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 8, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_filters)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(
        element::f32, PartialShape{1, 16, Dimension::dynamic(), Dimension::dynamic()});
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto output_shape = op::Constant::create(element::i64, Shape{2}, {3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(
        gcbd->get_output_partial_shape(0).same_scheme(PartialShape{1, Dimension::dynamic(), 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_shape_dyn_filters)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 8, 224, 224});
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_data_and_filters_1)
{
    // data batch shape: [N, C_IN * GROUPS, H, W]
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto output_shape = op::Constant::create(element::i64, Shape{3}, {3, 3, 3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().same_scheme(Rank{5}));
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3, 3}));
}

TEST(type_prop, group_conv_backprop_data_with_output_shape_dyn_data_and_filters_2)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto output_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, group_conv_backprop_data_dyn_data_and_filters)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto filters = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    ASSERT_TRUE(gcbd->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(gcbd->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, group_conv_backprop_data_invalid_element_types)
{
    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 8, 2, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f16, Shape{1, 16, 6, 6});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // data and filters should be of same element type
        FAIL() << "Incompatible element types not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Element types for data batch and filters do not match"));
    }
    catch (...)
    {
        FAIL() << "Element types validation check of inputs failed for unexpected reason";
    }

    try
    {
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
        const auto output_shape = op::Constant::create(element::f16, Shape{2}, {3, 3});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
        // output shape input element type must be of integer type
        FAIL() << "Incompatible element types not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element type for output shape should be of integer type");
    }
    catch (...)
    {
        FAIL() << "Element types validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, group_conv_backprop_data_invalid_input_ranks)
{
    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 6 (Conv3D)
    try
    {
        const auto filters = make_shared<op::Parameter>(
            element::f32, PartialShape{2, 8, 2, 3, 3, Dimension::dynamic()});
        const auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 16, 6, 6});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // data and weight have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for data batch and filters do not match."));
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    // data partial shape provided is rank 5 (Conv3D)
    // filter partial shape provided is rank 5 (Conv2D)
    try
    {
        const auto filters = make_shared<op::Parameter>(element::f32, PartialShape{2, 8, 2, 3, 3});
        const auto data = make_shared<op::Parameter>(
            element::f32, PartialShape{1, Dimension::dynamic(), 16, 6, 6});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // data and weight have incompatible ranks
        FAIL() << "Incompatible input ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for data batch and filters do not match."));
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }

    try
    {
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
        const auto output_shape = op::Constant::create(element::i64, Shape{2, 1}, {3, 3});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
        // Output shape optional input must be of rank 1
        FAIL() << "Incompatible output shape input rank not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Spatial shape of output input must be of rank 1"));
    }
    catch (...)
    {
        FAIL() << "Rank validation check of inputs failed for unexpected reason";
    }
}

TEST(type_prop, group_conv_backprop_data_invalid_params)
{
    try
    {
        // filter shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{21, 16, 20, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            filters,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1});
        // data batch shape does not have correct dimension C_IN * GROUPS
        FAIL() << "Incompatibile input shapes not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of data channels not a multiple of group size."));
    }
    catch (...)
    {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 16, 20, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            filters,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1});
        // filter shape specifies GROUPS = 4 and C_IN = 16, while data batch shape specifies
        // dimension C_IN * GROUPS = 16
        FAIL() << "Incompatibile input shapes not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data second dimension has incompatible value "
                                         "with number of input channels."));
    }
    catch (...)
    {
        FAIL() << "Input shapes validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 8, 2, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 6, 6});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, Strides{}, CoordinateDiff{1}, CoordinateDiff{1, 1}, Strides{});
        // pads_begin and pads_end do not match spatial dimensions
        FAIL() << "Incompatible pads number of spatial dimensions not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Pads should be defined for all and only spatial features.");
    }
    catch (...)
    {
        FAIL() << "Pads validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 4, 20, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, Strides{1}, CoordinateDiff{2, 2}, CoordinateDiff{2, 2}, Strides{1, 1});
        // Strides have incompatible number of spatial dimensions
        FAIL() << "Incompatible stride number of spatial dimensions not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Strides should be defined for all and only spatial features."));
    }
    catch (...)
    {
        FAIL() << "Strides validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 4, 20, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            filters,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1, 1});
        // Dilations have incompatible number of spatial dimensions
        FAIL() << "Incompatible dilations number of spatial dimensions not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dilations should be defined for all and only spatial features."));
    }
    catch (...)
    {
        FAIL() << "Dilations validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 4, 20, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            filters,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1},
                                                                            op::PadType::EXPLICIT,
                                                                            CoordinateDiff{0});
        // Output padding have incompatible number of spatial dimensions
        FAIL() << "Incompatible output padding number of spatial dimensions not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Output padding should be defined for all and only spatial features."));
    }
    catch (...)
    {
        FAIL() << "Output padding validation check failed for unexpected reason.";
    }

    try
    {
        // filters shape: [GROUPS, C_IN, C_OUT, kH, kW]
        const auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 3, 3});
        // data batch shape: [N, C_IN * GROUPS, H, W]
        const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 16, 5, 5});
        const auto output_shape = op::Constant::create(element::i64, Shape{3}, {3, 3, 3});
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, filters, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
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
