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

TEST(type_prop, asin_basic_shape_inference)
{
    Shape data_shape{2, 2};
    const auto param = make_shared<op::Parameter>(element::f32, data_shape);
    const auto op = make_shared<op::Asin>(param);
    ASSERT_EQ(op->get_shape(), (data_shape));
    ASSERT_EQ(op->get_element_type(), element::f32);
}

TEST(type_prop, asin_incompatible_input_type)
{
    Shape data_shape{3, 3};
    const auto param = make_shared<op::Parameter>(element::boolean, data_shape);
    ASSERT_THROW(make_shared<op::Asin>(param), ngraph::NodeValidationFailure);
}

TEST(type_prop, asin_dynamic_rank_input_2D)
{
    const PartialShape param_shape{Dimension::dynamic(), 10};
    const auto param = make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = make_shared<op::Asin>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape{Dimension(), 10}));
}

TEST(type_prop, asin_dynamic_rank_input_3D)
{
    const PartialShape param_shape{2, Dimension::dynamic(), Dimension::dynamic()};
    const auto param = make_shared<op::Parameter>(element::f32, param_shape);
    const auto op = make_shared<op::Asin>(param);
    ASSERT_TRUE(
        op->get_output_partial_shape(0).same_scheme(PartialShape{2, Dimension(), Dimension()}));
}

TEST(type_prop, asin_dynamic_rank_input_shape)
{
    const auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto op = make_shared<op::Asin>(param);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
