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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(type_prop, any_deduce)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    auto r0 = make_shared<op::Any>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Any>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Any>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Any>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, any_deduce_et_dynamic)
{
    auto param_0 = make_shared<op::Parameter>(element::dynamic, Shape{2, 4});

    auto r0 = make_shared<op::Any>(param_0, AxisSet{0});
    ASSERT_EQ(r0->get_element_type(), element::boolean);
    ASSERT_EQ(r0->get_shape(), (Shape{4}));

    auto r1 = make_shared<op::Any>(param_0, AxisSet{1});
    ASSERT_EQ(r1->get_element_type(), element::boolean);
    ASSERT_EQ(r1->get_shape(), (Shape{2}));

    auto r01 = make_shared<op::Any>(param_0, AxisSet{0, 1});
    ASSERT_EQ(r01->get_element_type(), element::boolean);
    ASSERT_EQ(r01->get_shape(), (Shape{}));

    auto r_none = make_shared<op::Any>(param_0, AxisSet{});
    ASSERT_EQ(r_none->get_element_type(), element::boolean);
    ASSERT_EQ(r_none->get_shape(), (Shape{2, 4}));
}

TEST(type_prop, any_et_non_boolean)
{
    auto param_0 = make_shared<op::Parameter>(element::i32, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Any>(param_0, AxisSet{0, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect invalid element type for Any";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element type must be boolean"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, any_axis_oob)
{
    auto param_0 = make_shared<op::Parameter>(element::boolean, Shape{2, 4});

    try
    {
        auto r = make_shared<op::Any>(param_0, AxisSet{0, 2, 1});
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for Any";
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

TEST(type_prop, any_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::boolean, PartialShape::dynamic());
    auto axes = AxisSet{2385, 0, 4404}; // arbitrary
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(any->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, any_partial_rank_static_dynamic_ok_result_static)
{
    auto param = make_shared<op::Parameter>(element::boolean,
                                            PartialShape{1, 2, Dimension::dynamic(), 4, 5});
    auto axes = AxisSet{2, 3};
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_EQ(any->get_shape(), (Shape{1, 2, 5}));
}

TEST(type_prop, any_partial_rank_static_dynamic_ok_result_dynamic)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 3};
    auto any = make_shared<op::Any>(param, axes);

    EXPECT_EQ(any->get_output_element_type(0), element::boolean);
    EXPECT_TRUE(
        any->get_output_partial_shape(0).same_scheme(PartialShape{1, 2, Dimension::dynamic()}));
}

TEST(type_prop, any_partial_rank_static_dynamic_axes_oob)
{
    auto param = make_shared<op::Parameter>(
        element::boolean, PartialShape{1, 2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto axes = AxisSet{2, 5, 1};

    try
    {
        auto any = make_shared<op::Any>(param, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Did not detect out-of-bound axis for Any (rank-static dynamic input)";
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
