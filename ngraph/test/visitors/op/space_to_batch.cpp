// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/space_to_batch.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, space_to_batch_op) {
    NodeBuilder::get_ops().register_factory<op::v1::SpaceToBatch>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});
    auto op = make_shared<op::v1::SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    NodeBuilder builder(op);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
