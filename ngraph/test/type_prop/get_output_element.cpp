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

TEST(type_prop, get_output_element_partial_et_dynamic)
{
    auto a = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto b = make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3, 4});
    auto add = make_shared<op::v1::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::dynamic);
    ASSERT_EQ(goe->get_output_shape(0), (Shape{1, 2, 3, 4}));
}

TEST(type_prop, get_output_element_partial_rank_dynamic)
{
    auto a = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto b = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    auto add = make_shared<op::v1::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::i32);
    ASSERT_TRUE(goe->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, get_output_element_partial_rank_static_dynamic)
{
    auto a = make_shared<op::Parameter>(
        element::i32, PartialShape{Dimension::dynamic(), 2, 3, Dimension::dynamic()});
    auto b = make_shared<op::Parameter>(
        element::i32, PartialShape{Dimension::dynamic(), 2, Dimension::dynamic(), 4});
    auto add = make_shared<op::v1::Add>(a, b);
    auto goe = make_shared<op::GetOutputElement>(add, 0);

    ASSERT_EQ(goe->get_output_element_type(0), element::i32);
    ASSERT_TRUE(
        goe->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic(), 2, 3, 4}));
}
