// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/depth_to_space.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, depth_to_space) {
    NodeBuilder::get_ops().register_factory<op::v0::DepthToSpace>();
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});

    const auto block_size = 2;
    const auto mode = "blocks_first";

    const auto dts = std::make_shared<op::v0::DepthToSpace>(data, mode, block_size);
    NodeBuilder builder(dts);
    auto g_dts = as_type_ptr<op::v0::DepthToSpace>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // depth_to_space attributes
    EXPECT_EQ(g_dts->get_block_size(), dts->get_block_size());
    EXPECT_EQ(g_dts->get_mode(), dts->get_mode());
}
