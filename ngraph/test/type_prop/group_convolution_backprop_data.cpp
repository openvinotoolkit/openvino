//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
    // GROUPS x C_IN x C_OUT x kH x kW
    const auto weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{2, 8, 2, 3, 3});
    // N x C_IN * GROUPS x H x W
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 6, 6});
    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, weights, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
    EXPECT_EQ(gcbd->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(gcbd->get_output_shape(0), (Shape{1, 4, 8, 8}));
    EXPECT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_pads_end(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::EXPLICIT);
}

TEST(type_prop, group_conv_backprop_data_output_shape)
{
    // N x C_IN * GROUPS x H x W
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 5, 5});
    // GROUPS x C_IN x C_OUT x kH x kW
    const auto weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 2, 3, 3});
    const auto output_shape = op::Constant::create(element::Type_t::i64, Shape{2}, {3, 3});

    const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, weights, output_shape, Strides{}, Strides{}, op::PadType::SAME_UPPER);
    EXPECT_EQ(gcbd->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(gcbd->get_output_shape(0), (Shape{1, 2, 3, 3}));
    EXPECT_EQ(gcbd->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(gcbd->get_pads_begin(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gcbd->get_pads_end(), (CoordinateDiff{2, 2}));
    EXPECT_EQ(gcbd->get_output_padding(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(gcbd->get_auto_pad(), op::PadType::SAME_UPPER);
}

TEST(type_prop, group_conv_bprop_data_v1_output_partial_shape_dynamic_static_rank)
{
    PartialShape shape_filter{4, 5, 2, 3, 3};
    auto filters = make_shared<op::Parameter>(element::Type_t::f32, shape_filter);
    PartialShape shape_data{Dimension(), 20, 224, 224};
    auto data = make_shared<op::Parameter>(element::Type_t::f32, shape_data);
    auto strides = Strides{2, 2};
    auto dilations = Strides{1, 1};
    auto padding_begin = CoordinateDiff{1, 1};
    auto padding_end = CoordinateDiff{1, 1};

    auto conv1 = make_shared<op::v1::GroupConvolutionBackpropData>(
        data, filters, strides, padding_begin, padding_end, dilations);

    ASSERT_TRUE(conv1->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(conv1->get_output_partial_shape(0).rank().same_scheme(Rank{4}));
    ASSERT_TRUE(conv1->get_output_partial_shape(0).is_dynamic());
    ASSERT_TRUE(conv1->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 8, 447, 447}));
}

TEST(type_prop, group_conv_backprop_data_invalid_params)
{
    // GROUPS x C_IN x C_OUT x kH x kW
    auto weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{21, 16, 20, 3, 3});
    // N x C_IN * GROUPS x H x W
    const auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 16, 5, 5});

    try
    {
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1});
        EXPECT_FALSE(gcbd.get()) << "GroupConvolutionBackpropData:v1 validation did not work. "
                                    "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Number of data channels not a multiple of group size."));
    }

    // GROUPS x C_IN x C_OUT x kH x kW
    weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{4, 16, 20, 3, 3});

    try
    {
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1});
        EXPECT_FALSE(gcbd.get()) << "GroupConvolutionBackpropData:v1 validation did not work. "
                                    "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data second dimension has incompatible value "
                                         "with number of input channels."));
    }

    // GROUPS x C_IN x C_OUT x kH x kW
    weights = make_shared<op::Parameter>(element::Type_t::f32, Shape{4, 4, 20, 3, 3});

    try
    {
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(
            data, weights, Strides{1}, CoordinateDiff{2, 2}, CoordinateDiff{2, 2}, Strides{1, 1});
        EXPECT_FALSE(gcbd.get()) << "GroupConvolutionBackpropData:v1 validation did not work. "
                                    "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Strides should be defined for all and only spatial features."));
    }

    try
    {
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1, 1});
        EXPECT_FALSE(gcbd.get()) << "GroupConvolutionBackpropData:v1 validation did not work. "
                                    "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dilations should be defined for all and only spatial features."));
    }

    try
    {
        const auto gcbd = make_shared<op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{1, 1},
                                                                            CoordinateDiff{2, 2},
                                                                            CoordinateDiff{2, 2},
                                                                            Strides{1, 1},
                                                                            op::PadType::EXPLICIT,
                                                                            CoordinateDiff{0});
        EXPECT_FALSE(gcbd.get()) << "GroupConvolutionBackpropData:v1 validation did not work. "
                                    "Node was created with incorrect params.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Output padding should be defined for all and only spatial features."));
    }
}
