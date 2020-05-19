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

TEST(type_prop, index_reduction_scalar)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 0, element::i32);
        FAIL() << "ArgMin c-tor should throw for scalar shapes";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Argument rank is zero");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_rank)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 2, element::i32);
        FAIL() << "ArgMin c-tor should throw for axis out of bounds";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axis (2) is not less than argument rank (2)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, argmin_invalid_zero_reduction_axis)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 1, element::i32);
        FAIL() << "ArgMin c-tor should throw for zero-length reduction axis";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "reduction axis can not be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, argmax_invalid_zero_reduction_axis)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, 1, element::i32);
        FAIL() << "ArgMax c-tor should throw for zero-length reduction axis";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "reduction axis can not be empty");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_invalid_index_type)
{
    auto a = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    try
    {
        auto argmin = make_shared<op::ArgMin>(a, 1, element::f32);
        FAIL() << "ArgMin c-tor should throw for invalid index type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_output_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::dynamic;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Invalid output type of element::dynamic not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_output_et_invalid)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::dynamic;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Invalid output type of element::f32 not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Index element is neither i64 or i32");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_dynamic_ok)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t axis = 228;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(argmax->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, index_reduction_partial_rank_static_dynamic_axis_oob)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 4;
    auto output_et = element::i32;

    try
    {
        auto argmax = make_shared<op::ArgMax>(a, axis, output_et);
        FAIL() << "Out-of-bounds reduction axis not detected (rank-static dynamic argument)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Reduction axis (4) is not less than argument rank (4)");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, index_reduction_partial_rank_static_dynamic_ok)
{
    auto a = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 2;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        argmax->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 4}));
}

TEST(type_prop, index_reduction_partial_et_dynamic_rank_static_dynamic_ok)
{
    auto a =
        make_shared<op::Parameter>(element::dynamic, PartialShape{Dimension::dynamic(), 2, 3, 4});
    size_t axis = 2;
    auto output_et = element::i32;

    auto argmax = make_shared<op::ArgMax>(a, axis, output_et);

    ASSERT_EQ(argmax->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        argmax->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 4}));
}
