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

TEST(type_prop, roll_output_shape_type_test)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 3, 4, 1, 5});
    auto shift = make_shared<op::Parameter>(element::i32, Shape{2});
    auto axes = make_shared<op::Parameter>(element::i64, Shape{2});

    auto r = make_shared<op::v7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{3, 3, 4, 1, 5}));
}

TEST(type_prop, roll_axis_scalar_test)
{
    auto arg = make_shared<op::Parameter>(element::i32, Shape{3, 3, 4});
    auto shift = make_shared<op::Parameter>(element::i64, Shape{1});
    auto axes = make_shared<op::Parameter>(element::i32, Shape{3});

    auto r = make_shared<op::v7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::i32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{3, 3, 4}));
}

TEST(type_prop, roll_invalid_axes_check)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 3, 4, 1, 5});
    auto shift = make_shared<op::Parameter>(element::i32, Shape{3});
    auto axes = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto r = make_shared<op::v7::Roll>(arg, shift, axes);
        // Should have thrown, so fail if it didn't
        FAIL() << "Unexpected pass with invalid axes and shift.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("If shift is a 1D vector, axes must be a 1D tensor of the same size."));
    }
    catch (...)
    {
        FAIL() << "Check failed for unexpected reason";
    }
}

TEST(type_prop, roll_dynamic_shape)
{
    auto arg = make_shared<op::Parameter>(element::f32,
                                          PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    auto shift = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    auto axes = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::v7::Roll>(arg, shift, axes);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(2)));
}
