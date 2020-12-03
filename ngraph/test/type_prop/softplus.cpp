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

TEST(type_prop, softplus)
{
    auto data = make_shared<op::Parameter>(element::Type_t::f32, Shape{1, 3, 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::Type_t::f32);
    EXPECT_EQ(softplus_func->get_shape(), (Shape{1, 3, 6}));
}

TEST(type_prop, softplus_partial)
{
    auto data =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));

    // rank unknown
    auto softplus_partial = make_shared<op::v4::SoftPlus>(
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic()));
    ASSERT_TRUE(softplus_partial->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, softplus_partial_static_rank)
{
    auto data =
        make_shared<op::Parameter>(element::Type_t::f32, PartialShape{1, Dimension::dynamic(), 6});
    auto softplus_func = make_shared<op::v4::SoftPlus>(data);
    EXPECT_EQ(softplus_func->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).same_scheme(
        (PartialShape{1, Dimension::dynamic(), 6})));
    ASSERT_TRUE(softplus_func->get_output_partial_shape(0).rank().is_static());
}
