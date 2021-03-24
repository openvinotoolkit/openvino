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

    auto dyn_indices = make_shared<op::Parameter>(element::i64, PartialShape{{1, 3}});
    auto dyn_ont_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    ASSERT_EQ(dyn_ont_hot->get_output_element_type(0), element::u32);
    ASSERT_EQ(dyn_ont_hot->get_output_partial_shape(0), (PartialShape{{1, 3}, 2}));
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

    auto dyn_indices = make_shared<op::Parameter>(element::i64, PartialShape{1, {3, 5}, 2, 3});
    auto dyn_ont_hot = make_shared<op::v1::OneHot>(dyn_indices, depth, on_value, off_value, axis);
    ASSERT_EQ(dyn_ont_hot->get_output_element_type(0), element::f32);
    ASSERT_EQ(dyn_ont_hot->get_output_partial_shape(0), (PartialShape{1, {3, 5}, 2, 4, 3}));
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

TEST(type_prop, one_hot_v1_out_types_1)
{
    auto indices = make_shared<op::Parameter>(element::i32, Shape{3, 2});
    auto depth = op::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = op::Constant::create(element::f32, Shape{}, {-3.3});
    auto off_value = op::Constant::create(element::f32, Shape{}, {-10.12});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::f32);
}

TEST(type_prop, one_hot_v1_out_types_2)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{3, 2});
    auto depth = op::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = op::Constant::create(element::i32, Shape{}, {-1});
    auto off_value = op::Constant::create(element::i32, Shape{}, {7});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::i32);
}

TEST(type_prop, one_hot_v1_out_types_3)
{
    auto indices = make_shared<op::Parameter>(element::i32, Shape{3, 2});
    auto depth = op::Constant::create(element::i32, Shape{}, {2});
    int64_t axis = -1;
    auto on_value = op::Constant::create(element::boolean, Shape{}, {true});
    auto off_value = op::Constant::create(element::boolean, Shape{}, {false});
    auto ont_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    ASSERT_EQ(ont_hot->get_element_type(), element::boolean);
}