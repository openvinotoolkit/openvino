// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, avg_pool_op) {
    NodeBuilder::get_ops().register_factory<opset1::AvgPool>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    bool exclude_pad = false;
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto avg_pool =
        make_shared<opset1::AvgPool>(data, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_mode, auto_pad);

    NodeBuilder builder(avg_pool);
    auto g_avg_pool = ov::as_type_ptr<opset1::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}

TEST(attributes, avg_pool_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::AvgPool>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 3, 37, 37});

    const auto strides = Strides{1, 1};
    const auto pads_begin = Shape{1, 1};
    const auto pads_end = Shape{1, 1};
    const auto kernel = Shape{2, 2};
    bool exclude_pad = false;
    const auto rounding_mode = op::RoundingType::CEIL;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto avg_pool =
        make_shared<opset8::AvgPool>(data, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_mode, auto_pad);
    NodeBuilder builder(avg_pool);
    auto g_avg_pool = ov::as_type_ptr<opset8::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}
