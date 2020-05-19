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

TEST(type_prop, random_uniform_et_mismatch)
{
    auto min_val = make_shared<op::Parameter>(element::f64, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Element type mismatch of min_val and max_val not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element types for min and max values do not match.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_et_non_float)
{
    auto min_val = make_shared<op::Parameter>(element::i64, Shape{});
    auto max_val = make_shared<op::Parameter>(element::i64, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-floating point element type not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element type of min_val and max_val inputs is not floating point.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_min_non_scalar)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{33});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-scalar min_val not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Tensor for min_value is not a scalar.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_max_non_scalar)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{44});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-scalar max_val not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Tensor for max_value is not a scalar.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_shape_non_i64)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::f64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-i64 result_shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Element type for result_shape is not element::i64.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_shape_non_scalar)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3, 3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-vector result_shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Tensor for result_shape not a vector.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_use_fixed_seed_non_boolean)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::i32, Shape{});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-boolean use_fixed_seed not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Element type for use_fixed_seed is not element::boolean.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_use_fixed_seed_non_scalar)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{3});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{1});
    size_t fixed_seed = 999;

    try
    {
        auto ru = make_shared<op::RandomUniform>(
            min_val, max_val, result_shape, use_fixed_seed, fixed_seed);
        FAIL() << "Non-scalar use_fixed_seed not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "Tensor for use_fixed_seed is not a scalar.");
    }
    catch (...)
    {
        FAIL() << "Test failed for unexpected reason";
    }
}

TEST(type_prop, random_uniform_use_result_shape_rank_dynamic)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    EXPECT_EQ(ru->get_output_element_type(0), element::f32);
    EXPECT_TRUE(ru->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, random_uniform_use_result_shape_rank_static_dynamic)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, PartialShape::dynamic(1));
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    EXPECT_EQ(ru->get_output_element_type(0), element::f32);
    EXPECT_TRUE(ru->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, random_uniform_use_result_shape_shape_static)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape = make_shared<op::Parameter>(element::i64, Shape{6});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    EXPECT_EQ(ru->get_output_element_type(0), element::f32);
    EXPECT_TRUE(ru->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(6)));
}

TEST(type_prop, random_uniform_use_result_shape_constant)
{
    auto min_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto max_val = make_shared<op::Parameter>(element::f32, Shape{});
    auto result_shape =
        make_shared<op::Constant>(element::i64, Shape{4}, std::vector<int64_t>{2, 4, 6, 9});
    auto use_fixed_seed = make_shared<op::Parameter>(element::boolean, Shape{});
    size_t fixed_seed = 999;

    auto ru =
        make_shared<op::RandomUniform>(min_val, max_val, result_shape, use_fixed_seed, fixed_seed);

    EXPECT_EQ(ru->get_output_element_type(0), element::f32);
    EXPECT_TRUE(ru->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 9}));
}
