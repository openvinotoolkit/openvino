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

TEST(type_prop, group_conv)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 91, 131}));
}

TEST(type_prop, group_conv_auto)
{
    // Deduce type
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 4, 100, 150});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 2, 10, 20});
    auto conv = make_shared<op::GroupConvolution>(data,
                                                  filters,
                                                  Strides{1, 1},
                                                  Strides{1, 1},
                                                  CoordinateDiff{0, 0},
                                                  CoordinateDiff{0, 0},
                                                  Strides{1, 1},
                                                  2,
                                                  op::PadType::AUTO);
    EXPECT_EQ(conv->get_shape(), (Shape{64, 128, 100, 150}));
    EXPECT_EQ(conv->get_padding_below(), (CoordinateDiff{4, 9}));
    EXPECT_EQ(conv->get_padding_above(), (CoordinateDiff{5, 10}));
}

TEST(type_prop, group_conv_invalid_groups)
{
    // Deduce type
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 20, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data channels not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{20, 10, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("# Filters not a multiple of group size"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
    try
    {
        auto conv = make_shared<op::GroupConvolution>(
            make_shared<op::Parameter>(element::f32, Shape{64, 30, 100, 150}),
            make_shared<op::Parameter>(element::f32, Shape{30, 20, 10, 20}),
            Strides{1, 1},
            Strides{1, 1},
            CoordinateDiff{0, 0},
            CoordinateDiff{0, 0},
            Strides{1, 1},
            3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid group conv";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incorrect number of channels per filter"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, group_conv_v1_partial_auto_padding_same_lower)
{
    const PartialShape data_batch_shape{1, 4, 5, 5};
    const PartialShape filters_shape{2, 1, 2, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_conv_v1_partial_auto_padding_same_upper)
{
    const PartialShape data_batch_shape{1, 4, 5, 5};
    const PartialShape filters_shape{2, 1, 2, 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_conv_v1_partial_auto_padding_same_lower_nc_dims_dynamic)
{
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{2, 1, 2, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 2, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{1, 1}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_conv_v1_partial_auto_padding_same_upper_nc_dims_dynamic)
{
    const PartialShape data_batch_shape{Dimension::dynamic(), Dimension::dynamic(), 5, 5};
    const PartialShape filters_shape{2, 1, 2, 2, 2};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_UPPER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 2, 5, 5}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{0, 0}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{1, 1}));
}

TEST(type_prop, group_conv_v1_partial_auto_padding_same_spatial_dims_dynamic)
{
    const PartialShape data_batch_shape{1, 4, Dimension::dynamic(), 5};
    const PartialShape filters_shape{2, 1, 2, 3, 3};
    Strides strides{1, 1};
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides dilations{1, 1};
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto data_batch = make_shared<op::Parameter>(element::f32, data_batch_shape);
    auto filters = make_shared<op::Parameter>(element::f32, filters_shape);

    auto conv = make_shared<op::v1::GroupConvolution>(
        data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad);

    ASSERT_TRUE(conv->get_output_partial_shape(0).same_scheme(
        {1, 2, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_EQ(conv->get_pads_begin(), (CoordinateDiff{}));
    ASSERT_EQ(conv->get_pads_end(), (CoordinateDiff{}));
}
