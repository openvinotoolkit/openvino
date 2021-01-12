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

using namespace std;
using namespace ngraph;

TEST(type_prop, abs_basic_shape_inference)
{
    Shape data_shape{2, 2};
    const auto param = make_shared<op::Parameter>(element::f32, data_shape);
    const auto op = make_shared<op::Abs>(param);
    ASSERT_EQ(op->get_shape(), (data_shape));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop, abs_incompatible_input_type)
{
    Shape data_shape{3, 3};
    const auto param = make_shared<op::Parameter>(element::boolean, data_shape);
    ASSERT_THROW(make_shared<op::Abs>(param), ngraph::NodeValidationFailure);
}

TEST(type_prop, abs_dynamic_shape_2D)
{
    const PartialShape data_shape{Dimension::dynamic(), 2};
    const auto param = make_shared<op::Parameter>(element::f32, data_shape);
    const auto op = make_shared<op::Abs>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 2}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop, abs_dynamic_shape_3D)
{
    const PartialShape data_shape{Dimension::dynamic(), Dimension::dynamic(), 3};
    const auto param = make_shared<op::Parameter>(element::f32, data_shape);
    const auto op = make_shared<op::Abs>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(
        {Dimension::dynamic(), Dimension::dynamic(), 3}));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop, abs_dynamic_ok)
{
    const auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto ap = make_shared<op::Abs>(param);
    ASSERT_EQ(ap->get_output_element_type(0), element::f32);
    ASSERT_TRUE(ap->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
