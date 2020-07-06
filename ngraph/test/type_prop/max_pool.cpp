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

TEST(type_prop, max_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{10});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(max_pool->get_window_shape(), Shape{2});
}

TEST(type_prop, max_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{10, 20}));
}

TEST(type_prop, max_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

    EXPECT_EQ(max_pool->get_element_type(), element::f32);
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(max_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(max_pool->get_window_shape(), (Shape{2, 3, 2}));
}

TEST(type_prop, max_pool_ceil_mode)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 10});
    Shape window_shape{2};
    auto move_strides = Strides{4};
    Shape padding_below{4};
    Shape padding_above{5};
    auto max_pool = make_shared<op::MaxPool>(param,
                                             window_shape,
                                             move_strides,
                                             padding_below,
                                             padding_above,
                                             op::PadType::EXPLICIT,
                                             true);

    // ceil((10 + 9 - 2)/4) + 1
    EXPECT_EQ(max_pool->get_shape(), (Shape{64, 3, 6}));
}

TEST(type_prop, max_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Data batch must have rank of at least 3"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Batch size is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count is zero"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0, 0, 0}), padding above "
                        "(CoordinateDiff{0, 0, 0}), window shape ({3,3,3}), and window strides "
                        "(Strides{1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0}), padding above "
                        "(CoordinateDiff{0}), window shape ({3}), and window strides (Strides{1}) "
                        "do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {6,2,10,10}, so data item "
                        "rank is 2), padding below (CoordinateDiff{0, 0}), padding above "
                        "(CoordinateDiff{0, 0}), window shape ({3,3}), and window strides "
                        "(Strides{2, 3, 8}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_input_data_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Data shape after padding and dilation has "
                                         "dimension less than 1 (dim: 0) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window after dilation has dimension less than 1 (dim: 0) at axis 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto max_pool = make_shared<op::MaxPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Window strides (Strides{0, 1}) has zero dimension at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, max_pool_partial_rank_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape ?, so data item rank is "
                        "?), padding below (CoordinateDiff{0, 0, 0, 0}), padding above "
                        "(CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5}), and window "
                        "strides (Strides{1, 1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic(6)};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 7, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5, 6};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Mismatch of attribute ranks not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Ranks for data item shape (data batch has shape {5,?,8,?,4,7}, so data "
                        "item rank is 4), padding below (CoordinateDiff{0, 0, 0, 0}), padding "
                        "above (CoordinateDiff{0, 0, 0, 0}), window shape ({2,3,4,5,6}), and "
                        "window strides (Strides{1, 1, 1, 1}) do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto mp = make_shared<op::MaxPool>(
            param, window_shape, window_movement_strides, padding_below, padding_above);
        FAIL() << "Oversized window not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Window after dilation has dimension (dim: 9) larger than "
                                         "the data shape after padding (dim: 8) at axis 0"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, max_pool_partial_rank_static_dynamic_padded_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{1, 0, 0, 0};

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::MaxPool>(
        param, window_shape, window_movement_strides, padding_below, padding_above);

    ASSERT_EQ(mp->get_output_element_type(0), element::f32);
    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, max_pool_auto_padding)
{
    const PartialShape arg_shape{1, 3, 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme({1, 3, 32, 32}));
}

TEST(type_prop, max_pool_auto_padding_nc_dims_dynamic)
{
    const PartialShape arg_shape{Dimension::dynamic(), Dimension::dynamic(), 32, 32};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        {Dimension::dynamic(), Dimension::dynamic(), 32, 32}));
}

TEST(type_prop, max_pool_auto_padding_spatial_dims_dynamic)
{
    const PartialShape arg_shape{1, 3, 32, Dimension::dynamic()};
    const Strides strides{1, 1};
    const Shape pads_begin{0, 0};
    const Shape pads_end{0, 0};
    const Shape kernel_shape{2, 2};
    const auto rounding_mode = op::RoundingType::FLOOR;
    const auto auto_pad = op::PadType::SAME_LOWER;

    auto arg = make_shared<op::Parameter>(element::f32, arg_shape);
    auto mp = make_shared<op::v1::MaxPool>(
        arg, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);

    ASSERT_TRUE(mp->get_output_partial_shape(0).same_scheme(
        {1, 3, Dimension::dynamic(), Dimension::dynamic()}));
}
