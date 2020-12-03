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

TEST(type_prop, mvn)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto mvn_func = make_shared<op::MVN>(data);
    EXPECT_EQ(mvn_func->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(mvn_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, mvn_partial)
{
    auto data =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto mvn_func = make_shared<op::MVN>(data);
    EXPECT_EQ(mvn_func->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(mvn_func->get_reduction_axes(), (AxisSet{1, 2}));
    ASSERT_TRUE(mvn_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // across_channels = false
    EXPECT_EQ(make_shared<op::MVN>(data, false)->get_reduction_axes(), (AxisSet{2}));

    // rank unknown
    auto mvn_partial = make_shared<op::MVN>(
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic()));
    EXPECT_EQ(mvn_partial->get_reduction_axes(), AxisSet{});
    ASSERT_TRUE(mvn_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
