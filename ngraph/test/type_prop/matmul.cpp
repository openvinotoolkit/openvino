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

TEST(type_prop, matmul_2D_same)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, matmul_4D_same)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 3}));
}

TEST(type_prop, matmul_2D)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2, 6, 4});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_5D_x_3D)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 1, 3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{7, 1, 5, 6, 4});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{7, 2, 5, 3, 4}));
}

TEST(type_prop, matmul_2D_transpose_a)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{6, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});

    auto matmul = make_shared<op::MatMul>(A, B, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D_transpose_a)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 6, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2, 6, 4});

    auto matmul = make_shared<op::MatMul>(A, B, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_2D_transpose_b)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4, 6});

    auto matmul = make_shared<op::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, matmul_4D_transpose_b)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 2, 4, 6});

    auto matmul = make_shared<op::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{2, 2, 3, 4}));
}

TEST(type_prop, matmul_dynamic_5D_transpose_b)
{
    Dimension dynamic = Dimension::dynamic();
    auto A =
        make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 4, dynamic, dynamic, 6});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{1, dynamic, dynamic, 4, 6});

    auto matmul = make_shared<op::MatMul>(A, B, 0, 1);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{dynamic, 4, dynamic, dynamic, 4}));
}

TEST(type_prop, matmul_dynamic_2D_transpose_a)
{
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{dynamic, 3});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{4, dynamic});

    auto matmul = make_shared<op::MatMul>(A, B, 1, 0);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{3, dynamic}));
}

TEST(type_prop, matmul_dynamic_1D_3D)
{
    Dimension dynamic = Dimension::dynamic();
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{dynamic});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, dynamic});

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), (PartialShape{2, dynamic}));
}

// Transpose attributes are ignored for 1D
TEST(type_prop, matmul_1D_x_1D_false_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_false_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<op::MatMul>(A, B, false, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_true_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<op::MatMul>(A, B, true, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_1D_x_1D_true_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1});

    auto matmul = make_shared<op::MatMul>(A, B, true, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{}));
}

TEST(type_prop, matmul_2D_x_1D_false_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_2D_x_1D_false_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2});

    auto matmul = make_shared<op::MatMul>(A, B, false, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_2D_x_1D_true_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2});
    auto matmul = make_shared<op::MatMul>(A, B, true, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_2D_x_1D_true_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2});

    auto matmul = make_shared<op::MatMul>(A, B, true, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_false_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_false_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    auto matmul = make_shared<op::MatMul>(A, B, false, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_true_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});
    auto matmul = make_shared<op::MatMul>(A, B, true, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}

TEST(type_prop, matmul_1D_x_2D_true_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    auto matmul = make_shared<op::MatMul>(A, B, true, true);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1}));
}
