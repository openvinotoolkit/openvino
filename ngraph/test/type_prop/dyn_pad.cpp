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

TEST(type_prop, dyn_pad_pad_value_test)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
    auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});

    // padding value matches tensor data-type
    try
    {
        auto pad_v = make_shared<op::Parameter>(element::i32, Shape{});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Padding value and arg type mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    // padding value is scalar
    try
    {
        auto pad_v = make_shared<op::Parameter>(element::f32, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "DynPad arg is not scalar");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_pad_wrong_ranks)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3, 4});
        auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Shape of padding below must be of rank 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
        auto pad_a = make_shared<op::Parameter>(
            element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Shape of padding above must be of rank 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
        auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Arg and padding below ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Arg and padding above ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }

    try
    {
        auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
        auto pad_b = make_shared<op::Parameter>(element::i64, Shape{4});
        auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
        auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Padding below and above ranks mismatch");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_pad_output_ranks_arg_static_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dyn_pad_output_ranks_arg_dynamic_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), 4, Dimension::dynamic()});
    auto pad_b = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto pad_a = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, dyn_pad_output_ranks_pad_static_ok)
{
    auto pad_v = make_shared<op::Parameter>(element::f32, Shape{});
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto pad_b = make_shared<op::Parameter>(element::i64, Shape{3});
    auto pad_a = make_shared<op::Parameter>(element::i64, Shape{3});
    auto dyn_pad = make_shared<op::DynPad>(arg, pad_b, pad_a, pad_v);

    EXPECT_EQ(dyn_pad->get_output_element_type(0), element::f32);
    EXPECT_TRUE(dyn_pad->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(3)));
}
