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

TEST(type_prop, transpose_arg_static_input_order_static_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_constant_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 1, 0, 3});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape{6, 4, 2, 8}));
}

TEST(type_prop, transpose_arg_static_input_order_constant_invalid_perm)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = op::Constant::create(element::i64, Shape{4}, vector<int64_t>{2, 9, 0, 3});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect invalid permutation";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Permutation AxisVector{2, 9, 0, 3} is not valid for input shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{Dimension::dynamic()});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_rank_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_static_input_order_static_input_order_wrong_size)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{5});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order wrong size";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input order must have shape [n], where n is the rank of arg."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_static_dynamic_input_order_static_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order = make_shared<op::Parameter>(element::i64, PartialShape{2, 2});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     transpose_arg_rank_static_dynamic_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(
        element::f32, PartialShape{2, Dimension::dynamic(), Dimension::dynamic(), 8});
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_arg_rank_dynamic_input_order_rank_static_dynamic_input_order_not_vector)
{
    auto arg = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto input_order =
        make_shared<op::Parameter>(element::i64, PartialShape{2, Dimension::dynamic()});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input order not vector";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input order must be a vector."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_input_order_et_dynamic_ok)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::dynamic, Shape{4});

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape::dynamic(4)));
}

TEST(type_prop, transpose_input_order_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto input_order = make_shared<op::Parameter>(element::boolean, Shape{4});

    try
    {
        auto r = make_shared<op::Transpose>(arg, input_order);
        FAIL() << "Did not detect input element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input order must have an integral number element type."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, transpose_with_empty_order)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 300});
    auto input_order = make_shared<op::Constant>(element::i64, Shape({0}), std::vector<size_t>());

    auto r = make_shared<op::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_TRUE(r->get_output_partial_shape(0).same_scheme(PartialShape({300, 1})));
}
