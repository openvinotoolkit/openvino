//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//  Licensed under the Apache License, Version 2.0 (the "License");
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

using namespace ngraph;

template <class T>
class UnaryOperator : public testing::Test
{
};

TYPED_TEST_CASE_P(UnaryOperator);

TYPED_TEST_P(UnaryOperator, basic_param_inference)
{
    {
        auto param = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
        auto op = std::make_shared<TypeParam>(param);
        ASSERT_EQ(op->get_shape(), (Shape{2, 2}));
        ASSERT_EQ(op->get_element_type(), element::f32);
    }
    {
        auto param = std::make_shared<op::Parameter>(element::i32, Shape{21, 15, 2});
        auto op = std::make_shared<TypeParam>(param);
        ASSERT_EQ(op->get_shape(), (Shape{21, 15, 2}));
        ASSERT_EQ(op->get_element_type(), element::i32);
    }
    {
        auto param = std::make_shared<op::Parameter>(element::u32, Shape{2, 10, 150});
        auto op = std::make_shared<TypeParam>(param);
        ASSERT_EQ(op->get_shape(), (Shape{2, 10, 150}));
        ASSERT_EQ(op->get_element_type(), element::u32);
    }
}

TYPED_TEST_P(UnaryOperator, incompatible_input_type)
{
    {
        const auto param = std::make_shared<op::Parameter>(element::boolean, Shape{100, 2, 50});
        ASSERT_THROW(std::make_shared<TypeParam>(param), ngraph::NodeValidationFailure);
    }
    {
        const auto param = std::make_shared<op::Parameter>(element::boolean, Shape{40, 17, 50});
        ASSERT_THROW(std::make_shared<TypeParam>(param), ngraph::NodeValidationFailure);
    }
}

TYPED_TEST_P(UnaryOperator, dynamic_rank_input_shape)
{
    {
        const auto param = std::make_shared<op::Parameter>(element::f64, PartialShape::dynamic());
        const auto op = std::make_shared<TypeParam>(param);
        ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    }
    {
        const auto param = std::make_shared<op::Parameter>(element::u32, PartialShape::dynamic());
        const auto op = std::make_shared<TypeParam>(param);
        ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    }
    {
        const auto param = std::make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
        const auto op = std::make_shared<TypeParam>(param);
        ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    }
}

REGISTER_TYPED_TEST_CASE_P(UnaryOperator,
                           basic_param_inference,
                           incompatible_input_type,
                           dynamic_rank_input_shape);

using MyTypes = ::testing::Types<op::Acos, op::Asin, op::Acosh>;

INSTANTIATE_TYPED_TEST_CASE_P(type_prop, UnaryOperator, MyTypes);