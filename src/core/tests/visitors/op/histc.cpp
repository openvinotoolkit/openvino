// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, histc_op) {
    NodeBuilder::opset().insert<ov::op::v17::Histc>();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    const auto histc = std::make_shared<ov::op::v17::Histc>(data, 7, -1.5, 3.25);

    NodeBuilder builder(histc, {data});
    const auto g_histc = ov::as_type_ptr<ov::op::v17::Histc>(builder.create());

    EXPECT_EQ(builder.get_value_map_size(), 3);
    EXPECT_EQ(g_histc->get_bins(), histc->get_bins());
    EXPECT_EQ(g_histc->get_min(), histc->get_min());
    EXPECT_EQ(g_histc->get_max(), histc->get_max());
}
