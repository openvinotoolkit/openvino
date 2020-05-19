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

TEST(type_prop, reshape_deduce_s2v)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::Reshape>(param, AxisVector{}, Shape{1, 1, 1});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_v2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_t2s)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{}));
}

TEST(type_prop, reshape_deduce_m2v_01)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_m2v_10)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{12});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_t2v_012)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{0, 1, 2}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_t2v_120)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{60});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_not_enough_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Not enough axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_too_many_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0, 3}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_duplicate_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 1, 0}, Shape{60});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{1, 2, 0}, Shape{3, 3, 3});
        // Should have thrown, so fail if it didn't
        FAIL() << "Too many axes not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Product of output shape dimensions does not match "
                                         "product of argument shape dimensions"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank dynamic, so we should set the desired output shape if the axis vector is not
// known invalid (invalid means it's not a permutation of {0,...,n-1} for any n).
//
TEST(type_prop, reshape_partial_rank_dynamic_axisvector_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 8, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

TEST(type_prop, reshape_partial_rank_dynamic_axisvector_not_ok)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 4}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect malformed AxisVector (input shape rank dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank static but input shape is dynamic, so should set desired output shape if the
// axis vector is consistent with the static rank.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_axisvector_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 8, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

TEST(type_prop, reshape_partial_rank_static_dynamic_axisvector_not_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect AxisVector inconsistent with rank (rank-static dynamic shape)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank static but input shape is dynamic, _but_ one of its static dimensions is zero,
// so should set desired output shape only if it also has zero elements.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0, 3}, Shape{3, 1, 0, 2});
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}

TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_not_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    try
    {
        auto r = make_shared<op::Reshape>(param, AxisVector{2, 1, 0}, Shape{3, 1, 8, 2});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect inconsistent output shape with static-zero-element rank-dynamic"
                  " static input shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input axis order is not a permutation of argument's axis indices"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
