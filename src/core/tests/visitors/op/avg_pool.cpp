// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;

TEST(attributes, avg_pool_op) {
    ov::test::NodeBuilder::opset().insert<op::v1::AvgPool>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    bool exclude_pad = false;
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto avg_pool =
        make_shared<op::v1::AvgPool>(data, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_mode, auto_pad);

    avg_pool->set_pads_begin(pads_begin);
    avg_pool->set_pads_end(pads_end);

    ov::test::NodeBuilder builder(avg_pool, {data});
    auto g_avg_pool = ov::as_type_ptr<op::v1::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}

TEST(attributes, avg_pool_op_valid) {
    ov::test::NodeBuilder::opset().insert<op::v1::AvgPool>();
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    bool exclude_pad = false;
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::VALID;

    auto avg_pool =
        make_shared<op::v1::AvgPool>(data, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_mode, auto_pad);

    ov::test::NodeBuilder builder(avg_pool, {data});
    auto g_avg_pool = ov::as_type_ptr<op::v1::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}

TEST(attributes, avg_pool_v8_op) {
    ov::test::NodeBuilder::opset().insert<op::v1::AvgPool>();
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{1, 3, 37, 37});

    const auto strides = Strides{1, 1};
    const auto pads_begin = Shape{1, 1};
    const auto pads_end = Shape{1, 1};
    const auto kernel = Shape{2, 2};
    bool exclude_pad = false;
    const auto rounding_mode = op::RoundingType::CEIL;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto avg_pool =
        make_shared<op::v1::AvgPool>(data, strides, pads_begin, pads_end, kernel, exclude_pad, rounding_mode, auto_pad);
    ov::test::NodeBuilder builder(avg_pool, {data});
    auto g_avg_pool = ov::as_type_ptr<op::v1::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}

TEST(attributes, avg_pool_v14_op) {
    ov::test::NodeBuilder::opset().insert<op::v14::AvgPool>();
    const auto data = make_shared<op::v0::Parameter>(element::i32, Shape{1, 3, 37, 37});

    const auto strides = Strides{1, 1};
    const auto pads_begin = Shape{1, 1};
    const auto pads_end = Shape{1, 1};
    const auto kernel = Shape{2, 2};
    bool exclude_pad = false;
    const auto rounding_mode = op::RoundingType::CEIL_TORCH;
    const auto auto_pad = op::PadType::EXPLICIT;

    const auto avg_pool = make_shared<op::v14::AvgPool>(data,
                                                        strides,
                                                        pads_begin,
                                                        pads_end,
                                                        kernel,
                                                        exclude_pad,
                                                        rounding_mode,
                                                        auto_pad);
    ov::test::NodeBuilder builder(avg_pool, {data});
    auto g_avg_pool = ov::as_type_ptr<op::v14::AvgPool>(builder.create());

    EXPECT_EQ(g_avg_pool->get_strides(), avg_pool->get_strides());
    EXPECT_EQ(g_avg_pool->get_pads_begin(), avg_pool->get_pads_begin());
    EXPECT_EQ(g_avg_pool->get_pads_end(), avg_pool->get_pads_end());
    EXPECT_EQ(g_avg_pool->get_kernel(), avg_pool->get_kernel());
    EXPECT_EQ(g_avg_pool->get_rounding_type(), avg_pool->get_rounding_type());
    EXPECT_EQ(g_avg_pool->get_auto_pad(), avg_pool->get_auto_pad());
}
