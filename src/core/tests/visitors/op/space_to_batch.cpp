// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_batch.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, space_to_batch_op) {
    NodeBuilder::opset().insert<ov::op::v1::SpaceToBatch>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});
    auto op = make_shared<ov::op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    NodeBuilder builder(op, {data, block_shape, pads_begin, pads_end});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v1::SpaceToBatch>(builder.create()));
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
