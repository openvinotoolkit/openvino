// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, adaptive_max_pool)
{
    const PartialShape arg_shape{1, 6, 8, 9};
    const vector<int64_t> output_shape{5, 7};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, output_shape);
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(0).same_scheme({1, 6, 5, 7}));
    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(1).same_scheme({1, 6, 5, 7}));
}

TEST(type_prop, adaptive_max_pool_i32_indices)
{
    const PartialShape arg_shape{1, 6, 8, 9};
    const vector<int64_t> output_shape{5, 7};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, output_shape);
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape, element::i32);

    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(0).same_scheme({1, 6, 5, 7}));
    ASSERT_EQ(adaptive_pool->output(1).get_element_type(), element::i32);
    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(1).same_scheme({1, 6, 5, 7}));
}

TEST(type_prop, adaptive_max_pool_dyn_batch)
{
    const PartialShape arg_shape{Dimension::dynamic(), 6, 8, 9};
    const vector<int64_t> output_shape{5, 7};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, output_shape);
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(
        adaptive_pool->get_output_partial_shape(0).same_scheme({Dimension::dynamic(), 6, 5, 7}));
    ASSERT_TRUE(
        adaptive_pool->get_output_partial_shape(1).same_scheme({Dimension::dynamic(), 6, 5, 7}));
}

TEST(type_prop, adaptive_max_pool_dyn_channels)
{
    const PartialShape arg_shape{1, Dimension::dynamic(), 8, 9};
    const vector<int64_t> output_shape{5, 7};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, output_shape);
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(
        adaptive_pool->get_output_partial_shape(0).same_scheme({1, Dimension::dynamic(), 5, 7}));
    ASSERT_TRUE(
        adaptive_pool->get_output_partial_shape(1).same_scheme({1, Dimension::dynamic(), 5, 7}));
}

TEST(type_prop, adaptive_max_pool_dyn_spatial)
{
    const PartialShape arg_shape{1, 6, Dimension::dynamic(), Dimension::dynamic()};
    const vector<int64_t> output_shape{5, 7};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, output_shape);
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(0).same_scheme({1, 6, 5, 7}));
    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(1).same_scheme({1, 6, 5, 7}));
}

TEST(type_prop, adaptive_max_pool_dyn_output_shape)
{
    const PartialShape arg_shape{1, 6, 8, 9};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(0).same_scheme(
        {1, 6, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(1).same_scheme(
        {1, 6, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, adaptive_max_pool_dyn_rank)
{
    const PartialShape arg_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = make_shared<op::Parameter>(element::i64, Shape{2});
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, out_shape);

    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    ASSERT_TRUE(adaptive_pool->get_output_partial_shape(1).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, adaptive_max_pool_unsupported_input_shape)
{
    const PartialShape arg_shape{1, 6};
    const vector<int64_t> output_shape{1};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{}, output_shape);

    EXPECT_THROW(make_shared<op::v8::AdaptiveMaxPool>(data, out_shape), NodeValidationFailure);
}

TEST(type_prop, adaptive_max_pool_wrong_out_shape)
{
    const PartialShape arg_shape{1, 6, 8, 9};
    const vector<int64_t> output_shape{5, 7, 8};

    auto data = make_shared<op::Parameter>(element::f32, arg_shape);
    auto out_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, output_shape);

    EXPECT_THROW(make_shared<op::v8::AdaptiveMaxPool>(data, out_shape), NodeValidationFailure);
}
