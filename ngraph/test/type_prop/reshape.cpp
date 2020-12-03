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

TEST(type_prop, reshape_deduce_s2v)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {1}, Shape{1}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1}));
}

TEST(type_prop, reshape_deduce_s2m)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {2}, Shape{1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1}));
}

TEST(type_prop, reshape_deduce_s2t)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {3}, Shape{1, 1, 1}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{1, 1, 1}));
}

TEST(type_prop, reshape_deduce_m2v_01)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {1}, Shape{12}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_m2v_10)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {1}, Shape{12}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{12}));
}

TEST(type_prop, reshape_deduce_t2v_012)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {1}, Shape{60}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_t2v_120)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {1}, Shape{60}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{60}));
}

TEST(type_prop, reshape_deduce_zero_special)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {3}, Shape{6, 2, 0}), true);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_EQ(r->get_shape(), (Shape{6, 2, 5}));
}

TEST(type_prop, reshape_deduce_wrong_output_shape)
{
    auto param = make_shared<op::Parameter>(element::Type_t::f32, Shape{3, 4, 5});
    try
    {
        auto r = make_shared<op::v1::Reshape>(
            param, op::Constant::create(element::Type_t::u64, {3}, Shape{3, 3, 3}), false);
        // Should have thrown, so fail if it didn't
        FAIL() << "No exception was thrown";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Check 'shape_size(get_input_shape(0)) == shape_size(output_shape)'"));
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
    auto param = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
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
    auto param = make_shared<op::Parameter>(element::Type_t::f32, param_shape);
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {4}, Shape{3, 1, 8, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
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
    auto param = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto r = make_shared<op::v1::Reshape>(
        param, op::Constant::create(element::Type_t::u64, {4}, Shape{3, 1, 0, 2}), false);
    ASSERT_EQ(r->get_element_type(), element::Type_t::f32);
    ASSERT_TRUE(r->get_output_partial_shape(0).is_static());
    ASSERT_EQ(r->get_shape(), (Shape{3, 1, 0, 2}));
}
