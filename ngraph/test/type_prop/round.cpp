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

TEST(type_prop, rounding_to_even)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    EXPECT_EQ(round_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, rounding_away)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto round_func =
        make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    EXPECT_EQ(round_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, rounding_to_even_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto round_partial = make_shared<op::v5::Round>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()),
        op::v5::Round::RoundMode::HALF_TO_EVEN);
    ASSERT_TRUE(round_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, rounding_away_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func =
        make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto round_partial = make_shared<op::v5::Round>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()),
        op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    ASSERT_TRUE(round_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, rounding_to_even_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func = make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_TO_EVEN);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(round_func->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, rounding_away_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto round_func =
        make_shared<op::v5::Round>(data, op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    EXPECT_EQ(round_func->get_element_type(), element::f32);
    ASSERT_TRUE(round_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(round_func->get_output_partial_shape(0).rank().is_static());
}
