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

TEST(type_prop, hsigmoid)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    EXPECT_EQ(hsigmoid_func->get_shape(), data->get_output_shape(0));
}

TEST(type_prop, hsigmoid_partial)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(
        hsigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));

    // rank unknown
    auto hsigmoid_partial = make_shared<op::v5::HSigmoid>(
        make_shared<op::Parameter>(element::f32, PartialShape::dynamic()));
    ASSERT_TRUE(hsigmoid_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, hsigmoid_partial_static_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto hsigmoid_func = make_shared<op::v5::HSigmoid>(data);
    EXPECT_EQ(hsigmoid_func->get_element_type(), element::f32);
    ASSERT_TRUE(
        hsigmoid_func->get_output_partial_shape(0).same_scheme(data->get_output_partial_shape(0)));
    ASSERT_TRUE(hsigmoid_func->get_output_partial_shape(0).rank().is_static());
}
