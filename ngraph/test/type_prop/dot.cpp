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

TEST(type_prop, dot_deduce_scalar_2d)
{
    // Deduce type for scalar/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_2d_scalar)
{
    // Deduce type for matrix/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 5}));
}

TEST(type_prop, dot_deduce_scalar_scalar)
{
    // Deduce type for scalar/scalar arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_scalar_1d)
{
    // Deduce type for scalar/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{6});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{6}));
}

TEST(type_prop, dot_deduce_1d)
{
    // Deduce type for vector/vector arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{}));
}

TEST(type_prop, dot_deduce_2d)
{
    // Deduce type for matrix/matrix arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{4, 3}));
}

TEST(type_prop, dot_deduce_different_rank)
{
    // Deduce type for different-rank tensor arguments
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 8, 4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3});
    auto bc = make_shared<op::Dot>(param1, param2);
    ASSERT_EQ(bc->get_element_type(), element::f32);
    ASSERT_EQ(bc->get_shape(), (Shape{2, 8, 4, 1, 3}));
}

TEST(type_prop, dot_deduce_element_type_mismatch)
{
    // Type deduction fails due to element type mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Element type mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Arguments do not have the same element type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_deduce_reduction_axes_size_mismatch)
{
    // Type deduction fails due to reduction axes size mismatch
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{3, 5});
    try
    {
        auto bc = make_shared<op::Dot>(param1, param2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dot reduction axes size mismatch not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Paired axes (axis 1 from arg0, axis 0 from arg1) do not have same length"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_both_rank_dynamic_axis_count_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_both_rank_dynamic_axis_count_explicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /*reduction axis count=*/1234);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_implicit)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_explicit_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dot_partial_left_rank_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL()
            << "Too many reduction axes not detected (rank-dynamic/rank-static dynamic operands)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_implicit)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_explicit_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_TRUE(d->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_dynamic_axis_count_explicit_too_many)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL()
            << "Too many reduction axes not detected (rank-dynamic/rank-static dynamic operands)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_implicit_1_ok)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 2});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 4, Dimension::dynamic(), 5}));
}

TEST(type_prop,
     dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_implicit_0_ok)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5});
    auto d = make_shared<op::Dot>(param0, param1);

    ASSERT_TRUE(d->get_output_partial_shape(0).same_scheme(
        PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic(), 5}));
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_left)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 5, 6});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_right)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 5, 6});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    dot_partial_left_rank_static_dynamic_right_rank_static_dynamic_axis_count_explicit_too_many_for_both)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
    try
    {
        auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/4);
        FAIL() << "Too many reduction axes not detected (rank-static dynamic/rank-static dynamic "
                  "operands)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axes count (4) is too large");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dot_partial_left_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::f32);
}

TEST(type_prop, dot_partial_right_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::i32);
}

TEST(type_prop, dot_partial_both_et_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::dynamic, PartialShape::dynamic());
    auto d = make_shared<op::Dot>(param0, param1, /* reduction axis count=*/3);

    ASSERT_EQ(d->get_output_element_type(0), element::dynamic);
}
