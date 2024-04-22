// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, scatter_nd_update) {
    NodeBuilder::opset().insert<ov::op::v3::ScatterNDUpdate>();

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1000, 256, 10, 15});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{25, 125, 3});
    auto updates = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{25, 125, 15});

    auto scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(data, indices, updates);
    NodeBuilder builder(scatter, {data, indices, updates});
    EXPECT_NO_THROW(auto g_scatter = ov::as_type_ptr<ov::op::v3::ScatterNDUpdate>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, scatter_nd_update_v15) {
    NodeBuilder::opset().insert<ov::op::v15::ScatterNDUpdate>();

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1000, 256, 10, 15});
    auto indices = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{25, 125, 3});
    auto updates = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{25, 125, 15});

    auto scatter = std::make_shared<ov::op::v15::ScatterNDUpdate>(data,
                                                                  indices,
                                                                  updates,
                                                                  op::v15::ScatterNDUpdate::Reduction::PROD);
    NodeBuilder builder(scatter, {data, indices, updates});
    const auto g_scatter = ov::as_type_ptr<ov::op::v15::ScatterNDUpdate>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_scatter->get_reduction(), scatter->get_reduction());
}
