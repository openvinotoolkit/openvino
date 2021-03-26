// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, max_pool_op)
{
    NodeBuilder::get_ops().register_factory<opset1::MaxPool>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto max_pool = make_shared<opset1::MaxPool>(
        data, strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);
    NodeBuilder builder(max_pool);
    auto g_max_pool = as_type_ptr<opset1::MaxPool>(builder.create());

    EXPECT_EQ(g_max_pool->get_strides(), max_pool->get_strides());
    EXPECT_EQ(g_max_pool->get_pads_begin(), max_pool->get_pads_begin());
    EXPECT_EQ(g_max_pool->get_pads_end(), max_pool->get_pads_end());
    EXPECT_EQ(g_max_pool->get_kernel(), max_pool->get_kernel());
    EXPECT_EQ(g_max_pool->get_rounding_type(), max_pool->get_rounding_type());
    EXPECT_EQ(g_max_pool->get_auto_pad(), max_pool->get_auto_pad());
}
