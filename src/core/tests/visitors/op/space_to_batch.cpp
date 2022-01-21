// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, space_to_batch_op) {
    using namespace opset2;

    NodeBuilder::get_ops().register_factory<SpaceToBatch>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 5});
    auto pads_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});
    auto op = make_shared<SpaceToBatch>(data, block_shape, pads_begin, pads_end);

    NodeBuilder builder(op);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
