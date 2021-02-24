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

TEST(type_prop, group_convolution_auto_padding_same_lower)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    element::Type_t et = element::f32;
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_TRUE(groupConv->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 5, 5}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_upper)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 2, 2};
    element::Type_t et = element::f32;
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    EXPECT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 5, 5}));
    EXPECT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_lower_spatial_dims_static)
{
    const PartialShape data_batch_pshape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{}, auto_pad);

    EXPECT_TRUE(groupConv->get_output_partial_shape(0).same_scheme(
        {Dimension::dynamic(), Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{1, 1}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_upper_spatial_dims_static)
{
    const PartialShape data_batch_pshape{1, Dimension::dynamic(), 5, 5};
    const PartialShape filters_pshape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2, 2};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{}, auto_pad);

    const auto output_pshape = groupConv->get_output_partial_shape(0);
    EXPECT_TRUE(output_pshape.same_scheme(PartialShape{1, Dimension::dynamic(), 5, 5}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_convolution_auto_padding_same_spatial_dims_dynamic)
{
    const PartialShape data_batch_pshape{1, 4, Dimension::dynamic(), 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    const element::Type_t et = element::f32;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{}, auto_pad);

    EXPECT_TRUE(
        groupConv->get_output_partial_shape(0).same_scheme({1, 2, Dimension::dynamic(), 5}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 1}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 1}));
}

TEST(type_prop, group_convolution_data_batch_dynamic)
{
    const PartialShape data_batch_pshape{PartialShape::dynamic()};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};
    const element::Type_t et = element::f32;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    EXPECT_EQ(groupConv->get_auto_pad(), op::PadType::EXPLICIT);
    EXPECT_EQ(groupConv->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(groupConv->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    const auto output_pshape = groupConv->get_output_partial_shape(0);
    EXPECT_TRUE(output_pshape.same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_convolution_filters_dynamic_auto_pad_explicit)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f16;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    EXPECT_EQ(groupConv->get_auto_pad(), op::PadType::EXPLICIT);
    EXPECT_EQ(groupConv->get_strides(), (Strides{1, 1}));
    EXPECT_EQ(groupConv->get_dilations(), (Strides{1, 1}));
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    const auto output_pshape = groupConv->get_output_partial_shape(0);
    ASSERT_TRUE(output_pshape.same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_convolution_filters_dynamic_auto_pad_same)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f16;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
    auto filters = make_shared<op::Parameter>(et, filters_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{}, auto_pad);

    EXPECT_EQ(groupConv->get_auto_pad(), op::PadType::SAME_LOWER);
    // pads should be as default since filters shape is dynamic
    EXPECT_EQ(groupConv->get_pads_begin(), (CoordinateDiff{0, 0}));
    EXPECT_EQ(groupConv->get_pads_end(), (CoordinateDiff{0, 0}));
    const auto output_pshape = groupConv->get_output_partial_shape(0);
    ASSERT_TRUE(output_pshape.same_scheme(
        PartialShape{1, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, group_convolution_data_batch_and_filters_dynamic)
{
    const PartialShape dyn_pshape{PartialShape::dynamic()};
    const element::Type_t et = element::f32;

    auto data_batch = make_shared<op::Parameter>(et, dyn_pshape);
    auto filters = make_shared<op::Parameter>(et, dyn_pshape);
    auto groupConv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});

    const auto output_pshape = groupConv->get_output_partial_shape(0);
    ASSERT_TRUE(output_pshape.same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, group_convolution_invalid_et_inputs)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 3, 3};

    try
    {
        auto data_batch = make_shared<op::Parameter>(element::f16, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(element::f32, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // data batch and filters must be of same element type
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
        const element::Type integral_et = element::u32;
        auto data_batch = make_shared<op::Parameter>(integral_et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(integral_et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, Strides{}, CoordinateDiff{}, CoordinateDiff{}, Strides{});
        // data batch and filters must be of float point element type
        FAIL() << "Integral element type of inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of inputs must be float point");
    }
    catch (...)
    {
        FAIL() << "Float element types of data batch and filters validation check failed for "
                  "unexpected reason.";
    }
}

TEST(type_prop, group_convolution_invalid_input_ranks)
{
    const element::Type_t et = element::f32;

    // data partial shape provided is rank 4 (Conv2D)
    // filter partial shape provided is rank 6 (Conv3D)
    try
    {
        auto filters =
            make_shared<op::Parameter>(et, PartialShape{2, 8, 2, 3, 3, Dimension::dynamic()});
        auto data = make_shared<op::Parameter>(et, PartialShape{1, 16, 6, 6});
        auto groupConv = make_shared<op::v1::GroupConvolution>(
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
        const auto filters = make_shared<op::Parameter>(et, PartialShape{2, 8, 2, 3, 3});
        const auto data =
            make_shared<op::Parameter>(et, PartialShape{1, Dimension::dynamic(), 16, 6, 6});
        const auto groupConv = make_shared<op::v1::GroupConvolution>(
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
}

TEST(type_prop, group_convolution_invalid_conv_param_spatial_dims)
{
    const PartialShape data_batch_pshape{1, 4, 5, 5};
    const PartialShape filters_pshape{2, 1, 2, 2, 2};
    const element::Type_t et = element::f32;

    // invalid strides spatial dimensions
    try
    {
        Strides strides{1, 1, 1};
        Strides dilations{1, 1};
        CoordinateDiff pads_begin{0, 0};
        CoordinateDiff pads_end{0, 0};

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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

        auto data_batch = make_shared<op::Parameter>(et, data_batch_pshape);
        auto filters = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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

        auto data_batch = make_shared<op::Parameter>(et, PartialShape::dynamic());
        auto filters = make_shared<op::Parameter>(et, filters_pshape);
        auto groupConv = make_shared<op::v1::GroupConvolution>(
            data_batch, filters, strides, pads_begin, pads_end, dilations);
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
}
