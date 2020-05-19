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

TEST(type_prop, scatter_nd_add_fail_indices_element_type)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i16, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
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

TEST(type_prop, scatter_nd_add_fail_indices_rank)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Indices rank is expected to be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_indices_last_dim)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{2, 4};
    Shape updates_shape{2, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices innermost dim";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Last dimension of indices can be at most the rank of inputs"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_element_type)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::i32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates element type";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Updates element type must be the same as inputs"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_rank)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Rank of updates must be rank of inputs + rank of indices "
                                         "- last dimension of indices - 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_add_fail_updates_shape)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{2, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterNDAdd>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect updates shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "updates_shape[indices_rank-1:] shape must be input_shape[indices_shape[-1]:]"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_fail_updates_element_type)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::i32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterND>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Created ScatterND op with incorrect updates element type.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates element type must be the same as element type of data."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_fail_updates_rank)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{1};
    Shape updates_shape{3, 3, 3};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterND>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Created ScatterND op with incorrect updates rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Updates rank is expected to be equal data_rank + indices_rank - "
                        "indices_shape[-1] - 1."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_nd_fail_updates_shape)
{
    Shape ref_shape{3, 3, 3};
    Shape indices_shape{4};
    Shape updates_shape{2};
    Shape out_shape{3, 3, 3};
    auto R = make_shared<op::Parameter>(element::f32, ref_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto U = make_shared<op::Parameter>(element::f32, updates_shape);
    try
    {
        auto G = make_shared<op::ScatterND>(R, I, U);
        // Should have thrown, so fail if it didn't
        FAIL() << "Created ScatterND op with incorrect indices shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Last dimension of indices can be at most the rank of data."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
