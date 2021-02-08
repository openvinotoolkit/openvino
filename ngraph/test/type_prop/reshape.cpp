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

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {2}, Shape{1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2m3)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {3}, Shape{1, 1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_2d_to_1d)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{12}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_3d_to_1d)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {1}, Shape{60}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_zero_special)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {3}, Shape{6, 2, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 2, 5}));
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::v1::Reshape>(
            param, op::Constant::create(element::u64, {3}, Shape{3, 3, 3}), false);
        // Should have thrown, so fail if it didn't
        FAIL() << "No exception was thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("is incompatible with input shape"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

//
// Input shape rank dynamic, so we should set the desired output shape
//
TEST(type_prop, reshape_partial_rank_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, so should set desired output shape
//
TEST(type_prop, reshape_partial_rank_static)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 6, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, param_shape);
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 8, 2}));
}

//
// Input shape rank static but input shape is dynamic, _but_ one of its static dimensions is zero,
// so should set desired output shape only if it also has zero elements.
//
TEST(type_prop, reshape_partial_rank_static_dynamic_but_zero_ok)
{
    auto param_shape =
        PartialShape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::u64, {4}, Shape{3, 1, 0, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{3, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{3, 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_shape(), (Shape{2, 2, 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension::dynamic(), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_copy_input_dynamic)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_zero)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{3, 1, 0, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{3, 1, Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_partial_rank_dynamic_special_neg)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{3, -1, 0, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), Dimension::dynamic(), 2}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3), 3});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3), 1, 3}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_zero_one_neg_double_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{2, Dimension(1, 3), Dimension::dynamic()});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {4}, std::vector<int64_t>{0, 0, 1, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0),
              (PartialShape{2, Dimension(1, 3), 1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_zero_neg_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{2, Dimension(1, 3)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{2, Dimension(1, 3)}));
}

TEST(type_prop, reshape_deduce_special_zero_shape_neg_zero_dynamic_with_interval_1)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{-1, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_pass_interval_dimension_through_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 2}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 3), 2}));
}

TEST(type_prop, reshape_multiply_interval_by_defined_dim_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), 2});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(2, 6)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_for_minus_one)
{
    auto param =
        make_shared<op::Parameter>(element::f32, PartialShape{1, Dimension(1, 3), Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18)}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_defined_dim_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32,
                                            PartialShape{1, Dimension(1, 3), 3, Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {3}, std::vector<int64_t>{0, -1, 3}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension(1, 18), 3}));
}

TEST(type_prop, reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, -1, Dimension(1, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop,
     reshape_multiply_interval_by_interval_divide_by_interval_for_minus_one_zero_included_in_input)
{
    auto param = make_shared<op::Parameter>(element::f32, PartialShape{1, -1, Dimension(0, 6)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {2}, std::vector<int64_t>{0, -1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{1, Dimension::dynamic()}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(1, 2), Dimension(1, 3), Dimension(1, 4)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(1, 24)}));
}

TEST(type_prop, reshape_multiply_intervals_by_interval_zero_included)
{
    auto param = make_shared<op::Parameter>(
        element::f32, PartialShape{Dimension(0, 2), Dimension(0, 3), Dimension(0, 4)});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::i64, {1}, std::vector<int64_t>{-1}), true);
    ASSERT_EQ(r->get_element_type(), element::f32);
    ASSERT_EQ(r->get_output_partial_shape(0), (PartialShape{Dimension(0, 24)}));
}
