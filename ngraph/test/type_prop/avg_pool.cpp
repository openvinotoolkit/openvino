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

TEST(type_prop, avg_pool_1d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{1});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100});
    Shape window_shape{10};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{10});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_uneven)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 2}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_1d_deduce_strided_small_even)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 6});
    Shape window_shape{2};
    auto move_strides = Strides{2};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), Strides{2});
    EXPECT_EQ(avg_pool->get_window_shape(), Shape{2});
    EXPECT_EQ(avg_pool->get_padding_below(), Shape{0});
    EXPECT_EQ(avg_pool->get_padding_above(), Shape{0});
}

TEST(type_prop, avg_pool_2d_deduce)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 91, 131}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{1, 1}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_2d_deduce_strided)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 100, 150});
    Shape window_shape{10, 20};
    auto move_strides = Strides{2, 3};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 46, 44}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{10, 20}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 3, 2, 3}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{0, 0, 0}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{0, 0, 0}));
}

TEST(type_prop, avg_pool_3d_deduce_strided_padded_small)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 7, 8, 10});
    Shape window_shape{2, 3, 2};
    auto move_strides = Strides{2, 3, 4};
    Shape padding_below{5, 6, 4};
    Shape padding_above{6, 4, 5};
    auto avg_pool = make_shared<op::AvgPool>(
        param, window_shape, move_strides, padding_below, padding_above, true);

    EXPECT_EQ(avg_pool->get_element_type(), element::f32);
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 9, 6, 5}));

    EXPECT_EQ(avg_pool->get_window_movement_strides(), (Strides{2, 3, 4}));
    EXPECT_EQ(avg_pool->get_window_shape(), (Shape{2, 3, 2}));
    EXPECT_EQ(avg_pool->get_padding_below(), (Shape{5, 6, 4}));
    EXPECT_EQ(avg_pool->get_padding_above(), (Shape{6, 4, 5}));
}

TEST(type_prop, avg_pool_ceil_mode)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{64, 3, 10});
    Shape window_shape{2};
    auto move_strides = Strides{4};
    Shape padding_below{4};
    Shape padding_above{5};
    auto avg_pool = make_shared<op::AvgPool>(param,
                                             window_shape,
                                             move_strides,
                                             padding_below,
                                             padding_above,
                                             true,
                                             op::PadType::EXPLICIT,
                                             true);

    // ceil((10 + 9 - 2)/4) + 1
    EXPECT_EQ(avg_pool->get_shape(), (Shape{64, 3, 6}));
}

TEST(type_prop, avg_pool_invalid_0d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 0D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_1d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 1D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_2d_input)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 6});
    Shape window_shape{};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid 2D input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Data batch must have rank of at least 3 (one batch axis, one "
                             "input-channel axis, and at least one spatial dimension)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_batch_size)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{0, 6, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 batch size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Batch size is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_0_channels)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 0, 1});
    Shape window_shape{1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0 channels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Channel count is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_many)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too many window dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0, 0, 0}), padding "
                             "above (CoordinateDiff{0, 0, 0}), window shape ({3,3,3}), and window "
                             "strides (Strides{1, 1, 1}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_wrong_number_of_window_dimensions_too_few)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with too few window dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0}), padding above "
                             "(CoordinateDiff{0}), window shape ({3}), and window strides "
                             "(Strides{1}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_movement_stride_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3, 8};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong movement stride rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{0, 0}), padding above "
                             "(CoordinateDiff{0, 0}), window shape ({3,3}), and window strides "
                             "(Strides{2, 3, 8}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_below_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2, 3};
    Shape padding_above{1, 2};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong below-padding rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{1, 2, 3}), padding "
                             "above (CoordinateDiff{1, 2}), window shape ({3,3}), and window "
                             "strides (Strides{2, 3}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_padding_above_rank)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{2, 3};
    Shape padding_below{1, 2};
    Shape padding_above{1, 2, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(
            param, window_shape, move_strides, padding_below, padding_above, false);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with wrong above-padding rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Ranks for data item shape (data batch has shape {6,2,10,10}, so data "
                             "item rank is 2), padding below (CoordinateDiff{1, 2}), padding above "
                             "(CoordinateDiff{1, 2, 3}), window shape ({3,3}), and window strides "
                             "(Strides{2, 3}) do not match");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_input_item_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 0, 10});
    Shape window_shape{3, 3};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length spatial axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            "Data shape after padding and dilation has dimension less than 1 (dim: 0) at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_window_size_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 0};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with zero-length window axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window after dilation has dimension less than 1 (dim: 0) at axis 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_invalid_dilated_too_large)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with oversized window not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window after dilation has dimension (dim: 9) larger than the data "
                             "shape after padding (dim: 8) at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_larger_than_pre_padding_but_fits_in_post_padding)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 8, 8});
    Shape window_shape{9, 9};
    Strides window_strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{1, 1};
    auto avg_pool =
        make_shared<op::AvgPool>(param, window_shape, window_strides, padding_below, padding_above);

    ASSERT_EQ(avg_pool->get_output_element_type(0), element::f32);
    ASSERT_EQ(avg_pool->get_output_shape(0), (Shape{6, 2, 1, 1}));
}

TEST(type_prop, avg_pool_invalid_movement_stride_0)
{
    // Deduce type
    auto param = make_shared<op::Parameter>(element::f32, Shape{6, 2, 10, 10});
    Shape window_shape{3, 3};
    auto move_strides = Strides{0, 1};
    try
    {
        auto avg_pool = make_shared<op::AvgPool>(param, window_shape, move_strides);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input with 0-length movement stride axis not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Window strides (Strides{0, 1}) has zero dimension at axis 0");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, avg_pool_partial_rank_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, avg_pool_partial_rank_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{PartialShape::dynamic()};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
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

TEST(type_prop, avg_pool_partial_rank_static_dynamic_ok)
{
    PartialShape arg_shape{PartialShape::dynamic(6)};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_some_dims_known_ok)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 7, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_attrib_rank_mismatch)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{2, 3, 4, 5, 6};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
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

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
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

TEST(type_prop, avg_pool_partial_rank_static_dynamic_padded_window_not_too_big)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 5};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 0};
    Shape padding_above{1, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);
    auto ap = make_shared<op::AvgPool>(param,
                                       window_shape,
                                       window_movement_strides,
                                       padding_below,
                                       padding_above,
                                       include_padding_in_average);

    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(
        PartialShape{5, Dimension::dynamic(), 1, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, avg_pool_partial_rank_static_dynamic_window_in_padding)
{
    PartialShape arg_shape{5, Dimension::dynamic(), 8, Dimension::dynamic(), 4, 7};
    Shape window_shape{9, 3, 4, 3};
    Strides window_movement_strides{1, 1, 1, 1};
    Shape padding_below{0, 0, 0, 4};
    Shape padding_above{0, 0, 0, 0};
    bool include_padding_in_average = false;

    auto param = make_shared<op::Parameter>(element::f32, arg_shape);

    try
    {
        auto ap = make_shared<op::AvgPool>(param,
                                           window_shape,
                                           window_movement_strides,
                                           padding_below,
                                           padding_above,
                                           include_padding_in_average);
        FAIL() << "Window in padding not detected";
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
