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

TEST(type_prop, gather_nd_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_batch_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, gather_nd_fail_params_rank)
{
    Shape params_shape{};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect params rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("params rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_rank)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("indices rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, gather_nd_fail_indices_element_type)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    try
    {
        auto G = make_shared<op::GatherND>(P, I);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices element type must be i64 or i32"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
