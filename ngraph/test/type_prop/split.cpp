// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, split_v1_axis_const_positive)
{
    const auto data = make_shared<op::Parameter>(element::f16, Shape{2, 3, 4});
    const auto axis = op::Constant::create(element::i64, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_element_type(i), element::f16);
        EXPECT_EQ(split->get_output_shape(i), (Shape{2, 1, 4}));
    }
}

TEST(type_prop, split_v1_axis_const_negative)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    const auto axis = op::Constant::create(element::i64, {}, {-2});
    const size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_element_type(i), element::i32);
        EXPECT_EQ(split->get_output_shape(i), (Shape{1, 6}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_known)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    const auto axis = op::Constant::create(element::i32, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), (PartialShape{2, 1, Dimension::dynamic()}));
    }
}

TEST(type_prop, split_v1_axis_const_only_data_axis_dim_known)
{
    const auto data = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic()});
    const auto axis = op::Constant::create(element::i16, {}, {0});
    const size_t num_splits = 2;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{1, Dimension::dynamic(), Dimension::dynamic()}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_unknown)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{4, Dimension::dynamic(), 3, 5});
    const auto axis = op::Constant::create(element::i8, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{4, Dimension::dynamic(), 3, 5}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_interval_known_divisible)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{4, Dimension(3, 6), 3, 5});
    const auto axis = op::Constant::create(element::i8, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{4, Dimension(1, 2), 3, 5}));
    }
}

TEST(type_prop, split_v1_axis_const_data_axis_dim_interval_known_upper_bound_divisible)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{4, Dimension(2, 4), 3, 5});
    const auto axis = op::Constant::create(element::i8, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i),
                  (PartialShape{4, Dimension(0, 1), 3, 5}));
    }
}

TEST(type_prop, split_v1_axis_const_invalid_data_axis_dim_interval_known)
{
    const auto data =
        make_shared<op::Parameter>(element::f32, PartialShape{4, Dimension(1, 2), 3, 5});
    const auto axis = op::Constant::create(element::i8, {}, {1});
    const size_t num_splits = 3;
    try
    {
        const auto split = make_shared<op::v1::Split>(data, axis, num_splits);
        FAIL() << "Invalid dimension of data input along axis not detected";
    }
    catch(const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
            "The interval maximum of the dimension for data input shape along 'axis' must be "
            "greater or equal to 'num_splits' attribute.");
    }
    catch (...)
    {
        FAIL() << "Invalid dimension of data input along axis validation check failed for unexpected reason";
    }
}

TEST(type_prop, split_v1_axis_const_only_data_rank_known)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = op::Constant::create(element::u64, {}, {1});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic(4));
    }
}

TEST(type_prop, split_v1_axis_param_only_data_rank_known)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic(4));
    const auto axis = make_shared<op::Parameter>(element::u32, PartialShape{});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic(4));
    }
}

TEST(type_prop, split_v1_axis_const_data_rank_unknown)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = op::Constant::create(element::u16, {}, {2});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}

TEST(type_prop, split_v1_axis_param_data_rank_unknown)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<op::Parameter>(element::u8, PartialShape{});
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}

TEST(type_prop, split_v1_axis_param_dynamic_ranks)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = make_shared<op::Parameter>(element::u8, PartialShape::dynamic());
    const size_t num_splits = 3;
    const auto split = make_shared<op::v1::Split>(data, axis, num_splits);

    EXPECT_EQ(split->outputs().size(), num_splits);
    for (size_t i = 0; i < num_splits; ++i)
    {
        EXPECT_EQ(split->get_output_partial_shape(i), PartialShape::dynamic());
    }
}

TEST(type_prop, split_v1_invalid_axis_et)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 6});

    try
    {
        auto axis = op::Constant::create(element::f32, Shape{}, {1});
        auto split = make_shared<op::v1::Split>(data, axis, 2);
        // axis input element type is floating-point
        FAIL() << "Invalid floating-point element type of 'axis' input not detected";
    }
    catch (NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of 'axis' input must be integer.");
    }
    catch(...)
    {
        FAIL() << "Element type of 'axis' input validation check failed for unexpected reason";
    }

    try
    {
        auto axis = op::Constant::create(element::boolean, Shape{}, {1});
        auto split = make_shared<op::v1::Split>(data, axis, 2);
        // axis input element type is boolean
        FAIL() << "Invalid boolean element type of axis input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type of 'axis' input must be integer.");
    }
    catch(...)
    {
        FAIL() << "Element type of 'axis' input validation check failed for unexpected reason";
    }
}

TEST(type_prop, split_v1_invalid_axis_not_a_scalar)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    auto axis = op::Constant::create(element::i64, Shape{2}, {0, 1});

    try
    {
        auto split = make_shared<op::v1::Split>(data, axis, 1);
        // axis has rank 1, not a scalar
        FAIL() << "Invalid shape of axis input not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("'axis' input must be a scalar."));
    }
    catch (...)
    {
        FAIL() << "Scalar 'axis' input validation check failed for unexpected reason";
    }
}

TEST(type_prop, split_v1_invalid_num_splits)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const size_t num_splits = 0;
    try
    {
        auto split = make_shared<op::v1::Split>(data, axis, num_splits);
        // num_splits value is zero
        FAIL() << "Invalid 'num_splits' attribute value not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Attribute 'num_splits' must be greater than zero");
    }
    catch(...)
    {
        FAIL() << "Attribute 'num_splits' validation check failed for unexpected reason";
    }
}

TEST(type_prop, split_v1_invalid_axis_value)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    auto axis = op::Constant::create(element::i64, Shape{}, {-5});
    const size_t num_splits = 4;
    try
    {
        auto split = make_shared<op::v1::Split>(data, axis, num_splits);
        // axis value not in the range [-2, 1]
        FAIL() << "Invalid 'axis' value not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Parameter axis -5 out of the tensor rank range");
    }
    catch(...)
    {
        FAIL() << "'axis' value validation check failed for unexpected reason";
    }
}

TEST(type_prop, split_v1_incompatible_data_shape_with_num_splits)
{
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});
    auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const size_t num_splits = 4;

    try
    {
        auto split = make_shared<op::v1::Split>(data, axis, num_splits);
        FAIL() << "Incompatible shape of data input along 'axis' and 'num_splits' attribute "
                  "not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Dimension of data input shape along 'axis': 6 must be evenly "
                        "divisible by 'num_splits' attribute value: 4"));
    }
    catch(...)
    {
        FAIL() << "Data input shape along 'axis' dimension validation check with "
                  "'num_splits' attribute, failed for unexpected reason";
    }
}
