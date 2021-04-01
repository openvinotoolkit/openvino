//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

TEST(type_prop, matmul_5D_x_3D_transpose_a_transpose_b)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{2, 1, 6, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{7, 1, 5, 4, 6});

    auto matmul = make_shared<op::MatMul>(A, B, true, true);

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
    ASSERT_EQ(matmul->get_output_partial_shape(0),
              (PartialShape{Dimension(1, -1), 4, dynamic, dynamic, 4}));
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
// 1D x 1D
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

TEST(type_prop, matmul_1D_x_1D_incompatible)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4});

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 2D x 1D
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

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B, true, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

TEST(type_prop, matmul_2D_x_1D_true_true)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B, true, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 1D x 2D
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

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B, false, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
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

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B, true, true);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible matrix dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul matrix dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

// 1D x 4D
TEST(type_prop, matmul_1D_x_4D_false_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1, 2, 4}));
}

// 4D x 1D
TEST(type_prop, matmul_4D_x_1D_false_false)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto B = make_shared<op::Parameter>(element::f32, Shape{4});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{1, 2, 3}));
}

// Batch broadcast
TEST(type_prop, matmul_batch_broadcast)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{5, 1, 1, 4, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1, 1, 6, 3, 2});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{5, 1, 6, 4, 2}));
}

TEST(type_prop, matmul_batch_broadcast_expand_to_A)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 4, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{7, 8, 5, 3, 2});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{7, 8, 5, 4, 2}));
}

TEST(type_prop, matmul_batch_broadcast_expand_to_B)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{8, 7, 6, 1, 4, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{1, 5, 3, 2});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_shape(), (Shape{8, 7, 6, 5, 4, 2}));
}

TEST(type_prop, matmul_incompatible_batch_dims)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{7, 4, 3});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 3, 2});

    try
    {
        auto matmul = make_shared<op::MatMul>(A, B);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible batch dimensions not detected. ";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Incompatible MatMul batch dimension"));
    }
    catch (...)
    {
        FAIL() << "MatMul shape validation failed for unexpected reason";
    }
}

TEST(type_prop, matmul_matrix_dynamic_bounds)
{
    auto A =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(2, 5), Dimension(6, 10)});
    auto B =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(7, 8), Dimension(15, 20)});

    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0),
              (PartialShape{Dimension(2, 5), Dimension(15, 20)}));
}

TEST(type_prop, matmul_batch_dynamic_bounds)
{
    // Input A and input B dim bounds => output dim bound
    // Dimension 1 can be expanded to any bigger

    Dimension dynamic = Dimension::dynamic();

    auto A_shape = PartialShape{dynamic,          // 0
                                Dimension(1, 5),  // 1
                                Dimension(2, 10), // 2
                                Dimension(5, 7),  // 3
                                Dimension(4, 7),  // 4
                                Dimension(5, 10), // 5
                                Dimension(1, 4),  // 6
                                Dimension(0, 1),  // 7
                                Dimension(0, 3),  // 8
                                1,                // 9
                                Dimension(1, -1), // 10
                                Dimension(1, 10), // 11
                                Dimension(2, -1), // 12
                                Dimension(1, -1), // 13
                                Dimension(2, -1), // 14
                                Dimension(1, -1), // 15
                                1,                // 16
                                1,                // 17
                                5,                // 18
                                6};               // 19

    auto B_shape = PartialShape{dynamic,           // 0
                                Dimension(10, 20), // 1
                                Dimension(10, 20), // 2
                                Dimension(4, 10),  // 3
                                Dimension(5, 10),  // 4
                                Dimension(4, 7),   // 5
                                dynamic,           // 6
                                Dimension(0, 1),   // 7
                                Dimension(2, 5),   // 8
                                Dimension(5, 10),  // 9
                                Dimension(1, 5),   // 10
                                Dimension(1, 5),   // 11
                                Dimension(1, 5),   // 12
                                Dimension(2, -1),  // 13
                                Dimension(2, -1),  // 14
                                Dimension(1, -1),  // 15
                                dynamic,           // 16
                                3,                 // 17
                                6,                 // 18
                                4};                // 19

    auto expected_output_shape = PartialShape{dynamic,           // 0
                                              Dimension(10, 20), // 1
                                              10,                // 2
                                              Dimension(5, 7),   // 3
                                              Dimension(5, 7),   // 4
                                              Dimension(5, 7),   // 5
                                              Dimension(1, -1),  // 6
                                              Dimension(0, 1),   // 7
                                              Dimension(2, 5),   // 8
                                              Dimension(5, 10),  // 9
                                              Dimension(1, -1),  // 10
                                              Dimension(1, 10),  // 11
                                              Dimension(2, -1),  // 12
                                              Dimension(2, -1),  // 13
                                              Dimension(2, -1),  // 14
                                              Dimension(1, -1),  // 15
                                              Dimension(1, -1),  // 16
                                              3,                 // 17
                                              5,                 // 18
                                              4};                // 19

    auto A = make_shared<op::Parameter>(element::f32, A_shape);
    auto B = make_shared<op::Parameter>(element::f32, B_shape);

    auto matmul = make_shared<op::MatMul>(A, B);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, matmul_incompatible_matrix_dim_bounds)
{
    auto A =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(2, 5), Dimension(3, 4)});
    auto B =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 2), Dimension(15, 20)});

    auto expected_output_shape = PartialShape{Dimension(2, 5), Dimension(15, 20)};

    // No error for backward compatibility
    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}

TEST(type_prop, matmul_incompatible_batch_dim_bounds)
{
    auto A = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(2, 5), 4, 3});
    auto B = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(6, 10), 3, 2});

    Dimension dynamic = Dimension::dynamic();
    auto expected_output_shape = PartialShape{dynamic, 4, 2};

    // No error for backward compatibility
    auto matmul = make_shared<op::MatMul>(A, B, false, false);

    ASSERT_EQ(matmul->get_element_type(), element::f32);
    ASSERT_EQ(matmul->get_output_partial_shape(0), expected_output_shape);
}
