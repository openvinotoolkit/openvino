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

TEST(type_prop, concat_deduce)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_deduce_wrong_rank)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32,
                                             Shape{
                                                 2,
                                                 2,
                                             });
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_wrong_shape)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_oob)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 5});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Concatenation axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_deduce_axis_barely_in_bounds)
{
    // Deduce type
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 8});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 12});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 2);
    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 3, 24}));
}

TEST(type_prop, concat_deduce_elem_type_mismatch)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::i32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Deduced type should disagree with specified type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_et_consistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{2, 12, 4}));
}

TEST(type_prop, concat_partial_et_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::dynamic, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::i32, Shape{2, 2, 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent element types not detected (some dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Argument element types are inconsistent"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_rank_dynamic)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(c->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_consistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_rank_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic(), 4});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent ranks not detected (some args rank-dynamic, some args rank-static "
                  "dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     concat_partial_some_rank_dynamic_others_rank_static_dynamic_dims_intransitively_inconsistent)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 3, Dimension::dynamic()});
    auto param3 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2, param3}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension::dynamic(), 3}));
}

TEST(type_prop,
     concat_partial_some_rank_dynamic_others_rank_static_with_concat_axis_static_dims_inconsistent)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});

    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_static)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_EQ(c->get_shape(), (Shape{2, 9, 3}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_compatible_result_dynamic)
{
    auto param0 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 2, Dimension::dynamic()});
    auto param1 = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic()});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{2, 3, Dimension::dynamic()});
    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);

    ASSERT_TRUE(
        c->get_output_partial_shape(0).same_scheme(PartialShape{2, 9, Dimension::dynamic()}));
}

TEST(type_prop, concat_partial_all_static_with_concat_axis_static_dims_incompatible)
{
    auto param0 = make_shared<op::Parameter>(element::f32, PartialShape{2, 2, 3});
    auto param1 =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4, 3});
    auto param2 =
        make_shared<op::Parameter>(element::f32, PartialShape{3, 3, Dimension::dynamic()});
    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, 1);
        // Should have thrown, so fail if it didn't
        FAIL() << "Inconsistent dimensions not detected (some args rank-dynamic, some args "
                  "rank-static dynamic)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shapes are inconsistent; they must have the same rank, and must "
                        "have equal dimension everywhere except on the concatenation axis"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, concat_partial_negative_axis_correct)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{3, 2, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{7, 2, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});

    auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, -3);

    ASSERT_EQ(c->get_element_type(), element::f32);
    ASSERT_EQ(c->get_shape(), (Shape{12, 2, 4}));
}

TEST(type_prop, concat_partial_negative_axis_incorrect)
{
    auto param0 = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto param1 = make_shared<op::Parameter>(element::f32, Shape{2, 7, 4});
    auto param2 = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4});

    try
    {
        auto c = make_shared<op::Concat>(NodeVector{param0, param1, param2}, -4);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect negative axis value not detected (out of bounds)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Concatenation axis (-1) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
