// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/specialize_function.hpp"

#include "gtest/gtest.h"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"

using namespace ov;
using namespace ov::test::utils;

using ov::Shape;
using ov::op::v0::Constant;
using ov::op::v0::Convert;
using ov::op::v0::Parameter;
using ov::op::v1::Add;

// Simple case: create a function with static parameter shapes and "specialize" them to the same
// shapes.
TEST(specialize_function, et_shape_static) {
    auto p0 = std::make_shared<Parameter>(element::f32, Shape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::i32, Shape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of dynamic element types.
TEST(specialize_function, et_dynamic_shape_static) {
    auto p0 = std::make_shared<Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-dynamic shapes.
TEST(specialize_function, et_static_shape_rank_dynamic) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of rank-static dynamic shapes.
TEST(specialize_function, et_static_shape_rank_static_dynamic) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic(3));
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic(3));

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);
}

// Test specialization of values to a shape-dynamic parameters.
TEST(specialize_function, et_static_shape_rank_static_dynamic_subst_val) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic(3));
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic(3));

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<int32_t> p1_subst_vals{5, 0, 3, 8, 5, 8};

    std::vector<void*> param_vals{nullptr, p1_subst_vals.data()};

    auto g = specialize_function(f,
                                 {element::f32, element::i32},
                                 {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                 param_vals);

    ASSERT_EQ(g->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(g->get_output_element_type(0), element::f32);

    auto plus_node = ov::as_type_ptr<Add>(g->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(plus_node);
    auto convert_node = ov::as_type_ptr<Convert>(plus_node->input_value(1).get_node_shared_ptr());
    ASSERT_TRUE(convert_node);
    auto const_node = ov::as_type_ptr<Constant>(convert_node->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(const_node);

    ASSERT_EQ(const_node->get_output_element_type(0), element::i32);
    ASSERT_EQ(const_node->get_output_shape(0), (Shape{1, 2, 3}));
    ASSERT_EQ(const_node->get_vector<int32_t>(), p1_subst_vals);
}

// Test specialization of rank-dynamic shapes to a case where validation will fail.
//
// (The input shapes we provide at specialization time are inconsistent.)
TEST(specialize_function, et_static_shape_rank_dynamic_validation_fails) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic());
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic());

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3, 4}},
                                param_vals);
        },
        ov::NodeValidationFailure);
}

// Test specialization of dynamic element types to a case where validation will fail.
//
// (The input element types we provide at specialization time are inconsistent.)
TEST(specialize_function, et_dynamic_shape_static_validation_fails) {
    auto p0 = std::make_shared<Parameter>(element::dynamic, Shape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::dynamic, Shape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::u32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                param_vals);
        },
        ov::NodeValidationFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have the wrong
// rank.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use OPENVINO_ASSERT, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_function, et_static_shape_rank_static_dynamic_rank_mismatch) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape::dynamic(3));
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape::dynamic(3));

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3, 4}},
                                param_vals);
        },
        ov::AssertFailure);
}

// Test specialization of rank-static dynamic shapes, where the replacement shapes have wrong
// dimensions.
//
// (Note that we are testing for a different exception class here because the failure is in
// specialize_shape's pre-checks, which use OPENVINO_ASSERT, rather than inside validation as we
// reconstruct the graph.)
TEST(specialize_function, et_static_shape_rank_static_dynamic_dim_mismatch) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape{1, ov::Dimension::dynamic(), 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 9, 4}},
                                param_vals);
        },
        ov::AssertFailure);
}

// Test for failure when we supply the wrong number of replacement element types.
TEST(specialize_function, et_count_wrong) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32, element::u32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                param_vals);
        },
        ov::AssertFailure);
}

// Test for failure when we supply the wrong number of replacement shapes.
TEST(specialize_function, shape_count_wrong) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}, ov::PartialShape{4, 5, 6}},
                                param_vals);
        },
        ov::AssertFailure);
}

// Test for failure when we supply the wrong number of replacement parameter values.
TEST(specialize_function, value_count_wrong) {
    auto p0 = std::make_shared<Parameter>(element::f32, ov::PartialShape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(element::i32, ov::PartialShape{1, 2, 3});

    auto k = std::make_shared<Convert>(p1, element::f32);
    auto a = std::make_shared<Add>(p0, k);

    auto f = std::make_shared<ov::Model>(a, ov::ParameterVector{p0, p1});

    std::vector<void*> param_vals{nullptr, nullptr, nullptr};

    ASSERT_THROW(
        {
            specialize_function(f,
                                {element::f32, element::i32},
                                {ov::PartialShape{1, 2, 3}, ov::PartialShape{1, 2, 3}},
                                param_vals);
        },
        ov::AssertFailure);
}
