//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <vector>
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

using namespace ngraph;

template <class T>
class ArithmeticOperator : public testing::Test
{
};

TYPED_TEST_SUITE_P(ArithmeticOperator);

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2, 3, 3}));
}

TYPED_TEST_P(ArithmeticOperator, default_autobroadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2}));
    ASSERT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, no_autobroadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 2});

    const auto op = std::make_shared<TypeParam>(A, B, op::AutoBroadcastSpec::NONE);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2}));
    ASSERT_EQ(op->get_autob(), op::AutoBroadcastType::NONE);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_scalar_numpy_broadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{1});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_1D_numpy_broadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{5});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_2D_x_4D_numpy_broadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_4D_numpy_broadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 4, 5});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{2, 3, 4, 5}));
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_4D_x_3D_numpy_broadcast)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{8, 1, 6, 1});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{7, 1, 5});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_shape(), (Shape{8, 7, 6, 5}));
    ASSERT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_element_types)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::i32, Shape{2, 2, 3, 3});

    ASSERT_THROW(std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, incompatible_boolean_type)
{
    auto A = std::make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});
    auto B = std::make_shared<op::Parameter>(element::boolean, Shape{2, 2, 3, 3});

    ASSERT_THROW(std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_1D_x_1D_incompatible)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{4});

    ASSERT_THROW(std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_3D_x_3D_incompatible)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{3, 5, 6});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{4, 10, 12});

    ASSERT_THROW(std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, shape_inference_5D_x_5D_incompatible)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{389, 112, 12});
    auto B = std::make_shared<op::Parameter>(element::f32, Shape{389, 112, 19});

    ASSERT_THROW(std::make_shared<TypeParam>(A, B), ngraph::NodeValidationFailure);
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_3D)
{
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});
    auto B = std::make_shared<op::Parameter>(element::f32, PartialShape{dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, dynamic_shape_5D)
{
    Dimension dynamic = Dimension::dynamic();
    auto A = std::make_shared<op::Parameter>(element::f32,
                                             PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B = std::make_shared<op::Parameter>(element::f32,
                                             PartialShape{dynamic, 4, dynamic, dynamic, 6});

    const auto op = std::make_shared<TypeParam>(A, B);

    ASSERT_EQ(op->get_element_type(), element::f32);
    ASSERT_EQ(op->get_output_partial_shape(0), (PartialShape{dynamic, 4, dynamic, dynamic, 6}));
}

TYPED_TEST_P(ArithmeticOperator, full_dynamic_shape)
{
    auto param = std::make_shared<op::Parameter>(element::f64, PartialShape::dynamic());
    const auto op = std::make_shared<TypeParam>(param, param);
    ASSERT_EQ(op->get_element_type(), element::f64);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

REGISTER_TYPED_TEST_SUITE_P(ArithmeticOperator,
                            shape_inference_2D,
                            shape_inference_4D,
                            default_autobroadcast,
                            no_autobroadcast,
                            shape_inference_4D_x_scalar_numpy_broadcast,
                            shape_inference_4D_x_1D_numpy_broadcast,
                            shape_inference_2D_x_4D_numpy_broadcast,
                            shape_inference_3D_x_4D_numpy_broadcast,
                            shape_inference_4D_x_3D_numpy_broadcast,
                            incompatible_element_types,
                            incompatible_boolean_type,
                            shape_inference_1D_x_1D_incompatible,
                            shape_inference_3D_x_3D_incompatible,
                            shape_inference_5D_x_5D_incompatible,
                            dynamic_shape_3D,
                            dynamic_shape_5D,
                            full_dynamic_shape);
