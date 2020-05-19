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

TEST(type_prop, sum_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    auto r0 = make_shared<op::Sum>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::f32);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Sum>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::f32);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Sum>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::f32);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Sum>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::f32);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, sum_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Sum>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for sum";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (2) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, sum_dynamic_axes)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto summation_axes = make_shared<op::Parameter>(element::i64, Shape{2});
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_TRUE(sum->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, sum_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto summation_axes = AxisSet{2385, 0, 4404}; // arbitrary
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_TRUE(sum->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, sum_partial_rank_static_dynamic_ok_result_static)
{
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto summation_axes = AxisSet{2, 3};
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_EQ(sum->get_shape(), (Shape{1, 2, 5}));
}

TEST(type_prop, sum_partial_rank_static_dynamic_ok_result_dynamic)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto summation_axes = AxisSet{2, 3};
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_TRUE(
        sum->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, sum_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto summation_axes = AxisSet{2, 5, 1};

    try
    {
        auto sum = make_shared<op::Sum>(param, summation_axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for sum (rank-static dynamic input)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis (5) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, sum_partial_negative_axes)
{
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto summation_axes = op::Constant::create(element::i64, Shape{2}, {-3, -2});
    auto sum = make_shared<op::Sum>(param, summation_axes);

    EXPECT_EQ(sum->get_output_element_type(0), element::f32);
    EXPECT_EQ(sum->get_shape(), (Shape{1, 2, 5}));
}
