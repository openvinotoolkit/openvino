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

TEST(type_prop, one_hot_deduce_scalar)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{});
    auto oh = make_shared<op::OneHot>(param, Shape{9}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9}));
}

TEST(type_prop, one_hot_deduce_vector_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{9, 8}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{9, 8}));
}

TEST(type_prop, one_hot_deduce_vector_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{8});
    auto oh = make_shared<op::OneHot>(param, Shape{8, 9}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{8, 9}));
}

TEST(type_prop, one_hot_deduce_matrix_0)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{2, 12, 24}, 0);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{2, 12, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_1)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 2, 24}, 1);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 2, 24}));
}

TEST(type_prop, one_hot_deduce_matrix_2)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 2}, 2);
    ASSERT_EQ(oh->get_element_type(), element::i32);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 2}));
}

TEST(type_prop, one_hot_deduce_et_dynamic)
{
    auto param = make_shared<op::Parameter>(element::dynamic, Shape{12, 24});
    auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 2}, 2);
    ASSERT_EQ(oh->get_element_type(), element::dynamic);
    ASSERT_EQ(oh->get_shape(), (Shape{12, 24, 2}));
}

TEST(type_prop, one_hot_deduce_floating_point)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid floating-point element type not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Argument does not have integral element type."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_axis_oob)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 24, 8}, 3);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("One-hot axis (3) is out of bounds"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_deduce_shape_incompatible)
{
    auto param = make_shared<op::Parameter>(element::i32, Shape{12, 24});
    try
    {
        auto oh = make_shared<op::OneHot>(param, Shape{12, 22, 8}, 2);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible one-hot output shape not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Argument shape {12,24} does not match the expected shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_dynamic)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{PartialShape::dynamic()};
    size_t one_hot_axis{3000};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic rank for requested result shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Requested result shape has dynamic rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_ok)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);

    ASSERT_EQ(oh->get_output_element_type(0), element::i32);
    ASSERT_TRUE(oh->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 2, 3, Dimension::dynamic()}));
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_one_hot_dim_dynamic)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{3};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic one-hot dimension not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,3,?}) has dynamic dimension "
                                         "at the one-hot axis (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_dynamic_rank_static_dynamic_one_hot_axis_oob)
{
    PartialShape input_shape{PartialShape::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic()};
    size_t one_hot_axis{4};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected (rank-dynamic argument, rank-static "
                  "dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("One-hot axis (4) is out of bounds (requested result shape: {?,2,3,?})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_ok)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);

    ASSERT_EQ(oh->get_output_element_type(0), element::i32);
    ASSERT_TRUE(oh->get_output_partial_shape(0).same_scheme(
        PartialShape{3, 2, 3, Dimension::dynamic(), 4}));
}

TEST(type_prop,
     one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_rank_input_short)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output ranks not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shape {3,?,?} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_rank_input_long)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4, 5};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output ranks not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Argument shape {3,?,?,4,5} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_incompatible_dim)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 5};
    PartialShape requested_shape{Dimension::dynamic(), 2, 3, Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible input/output dimensions not detected (rank-static dynamic "
                  "argument, rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Argument shape {3,?,?,5} does not match the expected shape of {?,2,?,4}"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_one_hot_dim_dynamic)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{
        Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Dynamic one-hot dimension not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,?,?,4}) has dynamic "
                                         "dimension at the one-hot axis (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_partial_rank_static_dynamic_rank_static_dynamic_one_hot_axis_oob)
{
    PartialShape input_shape{3, Dimension::dynamic(), Dimension::dynamic(), 4};
    PartialShape requested_shape{
        Dimension::dynamic(), 2, Dimension::dynamic(), Dimension::dynamic(), 4};
    size_t one_hot_axis{2};

    auto param = make_shared<op::Parameter>(element::i32, input_shape);
    try
    {
        auto oh = make_shared<op::OneHot>(param, requested_shape, one_hot_axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "One-hot axis out of bounds not detected (rank-static dynamic argument, "
                  "rank-static dynamic result shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Requested result shape ({?,2,?,?,4}) has dynamic "
                                         "dimension at the one-hot axis (2)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_output_shape)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{3});
    auto depth = op::Constant::create(element::i64, Shape{}, {2});
    auto on_value = op::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = -1;
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::u32);
    ASSERT_EQ(ont_hot->get_shape(), (Shape{3, 2}));
}

TEST(type_prop, one_hot_v1_output_shape_2)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = op::Constant::create(element::f32, Shape{}, {0.0f});
    int64_t axis = 3;
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::f32);
    ASSERT_EQ(ont_hot->get_shape(), (Shape{1, 3, 2, 4, 3}));
}

TEST(type_prop, one_hot_v1_indices_elem_not_integral)
{
    auto indices = make_shared<op::Parameter>(element::f16, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<op::Parameter>(element::u32, Shape{});
    auto off_value = make_shared<op::Parameter>(element::u32, Shape{});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Indices must be integral element type."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_depth_elem_not_integral)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::f16, Shape{});
    auto on_value = make_shared<op::Parameter>(element::u32, Shape{});
    auto off_value = make_shared<op::Parameter>(element::u32, Shape{});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect depth element type not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Depth must be integral element type."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_on_off_values_not_compatible)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<op::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<op::Parameter>(element::f16, Shape{});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incompatible on/off element types not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("on_value element type must be compatible with off_value element type."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_depth_not_scalar)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{1});
    auto on_value = make_shared<op::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<op::Parameter>(element::bf16, Shape{});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar depth input not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("depth input must be scalar."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_on_value_not_scalar)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<op::Parameter>(element::bf16, Shape{2});
    auto off_value = make_shared<op::Parameter>(element::bf16, Shape{});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar on_value input not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("on_value input must be scalar."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, one_hot_v1_off_value_not_scalar)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{2, 2});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{});
    auto on_value = make_shared<op::Parameter>(element::bf16, Shape{});
    auto off_value = make_shared<op::Parameter>(element::bf16, Shape{3});
    int64_t axis = -1;
    try
    {
        auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not scalar off_value input not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("off_value input must be scalar."));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
