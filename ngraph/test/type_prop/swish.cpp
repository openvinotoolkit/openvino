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

TEST(type_prop, swish)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(swish_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, swish_partial)
{
    auto data =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(
        swish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto swish_partial = make_shared<op::v4::Swish>(
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic()));
    ASSERT_TRUE(swish_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, swish_partial_static_rank)
{
    auto data =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto swish_func = make_shared<op::v4::Swish>(data);
    EXPECT_EQ(swish_func->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(
        swish_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, swish_incompatible_types)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::Type_t::f16, Shape{});
    try
    {
        const auto swish_func = make_shared<op::v4::Swish>(data, beta);
        FAIL() << "swish_func node was created with incompatible input data types.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Swish inputs must have the same type"));
    }
}

TEST(type_prop, swish_beta_not_scalar)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::Type_t::f32, Shape{1});
    try
    {
        const auto swish_func = make_shared<op::v4::Swish>(data, beta);
        FAIL() << "swish_func node was created with scalar beta value.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Swish input with beta must be scalar"));
    }
}

TEST(type_prop, swish_2_inputs)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto beta = make_shared<op::Parameter>(element::Type_t::f32, Shape{});
    const auto swish_func = make_shared<op::v4::Swish>(data, beta);

    EXPECT_EQ(swish_func->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).same_scheme(data->get_output_shape(0)));
    ASSERT_TRUE(swish_func->get_output_partial_shape(0).rank().is_static());
}
