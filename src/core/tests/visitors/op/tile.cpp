// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, tile_op) {
    NodeBuilder::opset().insert<ov::op::v0::Tile>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto repeats = make_shared<ov::op::v0::Constant>(element::i64, Shape{4});

    const auto tile = make_shared<op::v0::Tile>(data, repeats);
    NodeBuilder builder(tile, {data, repeats});
    EXPECT_NO_THROW(auto g_tile = ov::as_type_ptr<op::v0::Tile>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
