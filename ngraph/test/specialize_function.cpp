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
#include "ngraph/specialize_function.hpp"

using namespace ngraph;

// Simple case: create a function with static parameter shapes and "specialize" them to the same
// shapes.
TEST(specialize_function, et_shape_static)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of dynamic element types.
TEST(specialize_function, et_dynamic_shape_static)
{
    auto p0 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-dynamic shapes.
TEST(specialize_function, et_static_shape_rank_dynamic)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-static dynamic shapes.
TEST(specialize_function, et_static_shape_rank_static_dynamic)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic(3));

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of values to a shape-dynamic parameters.
TEST(specialize_function, et_static_shape_rank_static_dynamic_subst_val)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic(3));

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<int32_t> p1_subst_vals{5, 0, 3, 8, 5, 8};

    std::vector<void*> param_vals{nullptr, p1_subst_vals.data()};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);

    auto plus_node = as_type_ptr<op::v1::Add>(g->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(plus_node);
    auto convert_node = as_type_ptr<op::Convert>(plus_node->get_argument(1));
    ASSERT_TRUE(convert_node);
    auto const_node = as_type_ptr<op::Constant>(convert_node->get_argument(0));
    ASSERT_TRUE(const_node);

    ASSERT_EQ(const_node->get_output_element_type(0), element::i32);
    ASSERT_EQ(const_node->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(const_node->get_vector<int32_t>(), p1_subst_vals);
}

// Test specialization of rank-dynamic shapes to a case where validation will fail.
//
// (The input shapes we provide at specialization time are inconsistent.)
TEST(specialize_function, et_static_shape_rank_dynamic_validation_fails)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic());

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3, 4}},
                                param_vals);
        },
        NodeValidationFailure);
}

// Test specialization of dynamic element types to a case where validation will fail.
//
// (The input element types we provide at specialization time are inconsistent.)
TEST(specialize_function, et_dynamic_shape_static_validation_fails)
{
    auto p0 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::u32, element::i32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                param_vals);
        },
        NodeValidationFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have the wrong
// rank.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use NGRAPH_CHECK, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_function, et_static_shape_rank_static_dynamic_rank_mismatch)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic(3));
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape::dynamic(3));

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3, 4}},
                                param_vals);
        },
        CheckFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have wrong
// dimensions.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use NGRAPH_CHECK, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_function, et_static_shape_rank_static_dynamic_dim_mismatch)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 =
        std::make_shared<op::Parameter>(element::i32, PartialShape{1, Dimension::dynamic(), 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 9, 4}},
                                param_vals);
        },
        CheckFailure);
}

// Test for failure when we supply the wrong number of replacement element types.
TEST(specialize_function, et_count_wrong)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32, element::u32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                param_vals);
        },
        CheckFailure);
}

// Test for failure when we supply the wrong number of replacement shapes.
TEST(specialize_function, shape_count_wrong)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(
                f,
                {element::f32, element::i32},
                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}, PartialShape{4, 5, 6}},
                param_vals);
        },
        CheckFailure);
}

// Test for failure when we supply the wrong number of replacement parameter values.
TEST(specialize_function, value_count_wrong)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3});
    auto p1 = std::make_shared<op::Parameter>(element::i32, PartialShape{1, 2, 3});

    auto k = std::make_shared<op::Convert>(p1, element::f32);
    auto a = p0 + k;

    auto f = std::make_shared<Function>(a, ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {PartialShape{1, 2, 3}, PartialShape{1, 2, 3}},
                                param_vals);
        },
        CheckFailure);
}

// Test checks that constant sharing is working
TEST(specialize_function, share_constants)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 64, 64});
    auto mul_const = op::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1, 2, 3});
    auto mul = std::make_shared<op::Multiply>(p0, mul_const, op::AutoBroadcastType::NUMPY);

    auto add_const = op::Constant::create(element::f32, Shape{1, 3, 1, 1}, {4, 5, 6});
    auto add = std::make_shared<op::Multiply>(mul, add_const, op::AutoBroadcastType::NUMPY);

    auto f = std::make_shared<Function>(add, ParameterVector{p0});

    auto f_specialized =
        specialize_function(f, {element::f32}, {PartialShape{2, 3, 64, 64}}, {nullptr}, true, true);

    ASSERT_EQ(mul_const->get_output_target_inputs(0).size(), 2);
    ASSERT_EQ(add_const->get_output_target_inputs(0).size(), 2);
}

// Test checks that constant sharing works when constant folding replaces constants
TEST(specialize_function, share_constants_with_cf)
{
    auto p0 = std::make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 64, 64});
    auto mul_const = op::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1, 2, 3});
    auto mul = std::make_shared<op::Multiply>(p0, mul_const, op::AutoBroadcastType::NUMPY);

    auto add_const_1 = op::Constant::create(element::f32, Shape{1, 3, 1, 1}, {4, 5, 6});
    auto add_const_2 = op::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1, 2, 3});
    auto add_const = std::make_shared<op::v1::Add>(add_const_1, add_const_2);
    auto add = std::make_shared<op::v1::Add>(mul, add_const, op::AutoBroadcastType::NUMPY);

    auto f = std::make_shared<Function>(add, ParameterVector{p0});

    auto f_specialized =
        specialize_function(f, {element::f32}, {PartialShape{2, 3, 64, 64}}, {nullptr}, true, true);

    ASSERT_EQ(mul_const->get_output_target_inputs(0).size(), 2);
    ASSERT_EQ(add_const_1->get_output_target_inputs(0).size(), 1);
    ASSERT_EQ(add_const_2->get_output_target_inputs(0).size(), 1);
}
