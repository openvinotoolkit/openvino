// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, histc_v17_op_defaults) {
    NodeBuilder::opset().insert<ov::op::v17::Histc>();
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{10});
    auto histc = std::make_shared<ov::op::v17::Histc>(data);
    NodeBuilder builder(histc, {data});

    auto g_histc = ov::as_type_ptr<ov::op::v17::Histc>(builder.create());
    EXPECT_EQ(g_histc->get_bins(), histc->get_bins());
    EXPECT_DOUBLE_EQ(g_histc->get_min_val(), histc->get_min_val());
    EXPECT_DOUBLE_EQ(g_histc->get_max_val(), histc->get_max_val());
}

TEST(attributes, histc_v17_op_custom_attributes) {
    NodeBuilder::opset().insert<ov::op::v17::Histc>();
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f64, Shape{10});
    auto histc = std::make_shared<ov::op::v17::Histc>(data, int64_t{17}, -2.5, 6.5);
    NodeBuilder builder(histc, {data});

    auto g_histc = ov::as_type_ptr<ov::op::v17::Histc>(builder.create());
    EXPECT_EQ(g_histc->get_bins(), 17);
    EXPECT_DOUBLE_EQ(g_histc->get_min_val(), -2.5);
    EXPECT_DOUBLE_EQ(g_histc->get_max_val(), 6.5);
}
