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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

// ------------------------------ V1 ------------------------------

TEST(type_prop, gather_axis_0)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
    ASSERT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_axis_1)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {1});
    auto G = make_shared<op::v1::Gather>(P, I, A);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
    ASSERT_EQ(G->get_axis(), 1);
}

TEST(type_prop, gather_v1_incorrect_axis_shape)
{
    auto params = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<op::Parameter>(element::i64, Shape{2});
    try
    {
        auto G = make_shared<op::v1::Gather>(params, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect axis input shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Axes input must be scalar or have 1 element (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_v1_axis_out_of_input_rank)
{
    auto params = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    try
    {
        auto G = make_shared<op::v1::Gather>(params, indices, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect element of axis input";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The axis must => 0 and <= input_rank (axis:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_v1_negative_axis)
{
    auto params = make_shared<op::Parameter>(element::f32, Shape{5, 6, 7});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    int64_t axis = -2;
    auto axis_node = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto gather_v1 = make_shared<op::v1::Gather>(params, indices, axis_node);
    ASSERT_EQ(gather_v1->get_axis(), 1);
}

// ------------------------------ V7 ------------------------------

TEST(type_prop, gather_7_axis_0)
{
    PartialShape data_shape{3, 2};
    PartialShape indices_shape{2, 2};
    PartialShape out_shape{2, 2, 2};
    int64_t batch_dims = 0;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {0});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
    ASSERT_EQ(G->get_axis(), 0);
}

TEST(type_prop, gather_7_axis_1)
{
    PartialShape data_shape{3, 3};
    PartialShape indices_shape{1, 2};
    PartialShape out_shape{3, 1, 2};
    int64_t axis = 1;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto A = op::Constant::create(element::i64, Shape{}, {axis});
    auto G = make_shared<op::v7::Gather>(D, I, A);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
    ASSERT_EQ(G->get_axis(), 1);
}

TEST(type_prop, gather_7_negative_axis)
{
    PartialShape data_shape{5, 6, 7};
    PartialShape indices_shape{4};
    PartialShape out_shape{5, 4, 7};
    int64_t axis = -2;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A);

    ASSERT_EQ(G->get_axis(), 1);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_batch_dims_1_axis_3)
{
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(1, 3), 200, Dimension(2, 10), 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 1;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_batch_dim)
{
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(7, 10), 3, 8};
    PartialShape out_shape{7, 3, 8, 20};
    int64_t axis = 1;
    int64_t batch_dims = 1;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_2d_batch_dim)
{
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 3, 8, 400};
    int64_t axis = 2;
    int64_t batch_dims = 2;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_2d_batch_dim_axis_3)
{
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape{Dimension(7, 10), Dimension(2, 10), 3, 8};
    PartialShape out_shape{7, Dimension(2, 3), 200, 3, 8};
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

TEST(type_prop, gather_7_dynamic_data_indices_rank)
{
    PartialShape data_shape{Dimension(1, 7), Dimension(1, 3), 200, 400};
    PartialShape indices_shape = PartialShape::dynamic();
    PartialShape out_shape = PartialShape::dynamic();
    int64_t axis = 3;
    int64_t batch_dims = 2;

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);

    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_output_partial_shape(0), out_shape);
}

// --------------------- Negative tests ------------------------------

TEST(type_prop, gather_7_incorrect_axis_shape)
{
    auto D = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<op::Parameter>(element::i64, Shape{4});
    auto A = make_shared<op::Parameter>(element::i64, Shape{2});

    try
    {
        auto G = make_shared<op::v7::Gather>(D, I, A);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect A input shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Axes input must be scalar or have 1 element (shape:"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_7_axis_out_of_input_rank)
{
    auto D = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto I = make_shared<op::Parameter>(element::i64, Shape{4});
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{2});
    int64_t batch_dims = 0;
    try
    {
        auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect element of A input";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("The axis must be => 0 and < data_rank. But instead got"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_7_dynamic_batch_dims_inconsistent)
{
    PartialShape data_shape{Dimension(1, 7), 20, 20};
    PartialShape indices_shape{Dimension(8, 10), 3, 8};

    auto D = make_shared<op::Parameter>(element::f32, data_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    int64_t axis = 1;
    auto A = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{axis});
    int64_t batch_dims = 1;

    try
    {
        auto G = make_shared<op::v7::Gather>(D, I, A, batch_dims);
        // Should have thrown, so fail if it didn't
        FAIL() << "Shape inconsistency check for dynamic PartialShape failed";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("data and indices must have equal or intersecting sizes until batch_dims"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}