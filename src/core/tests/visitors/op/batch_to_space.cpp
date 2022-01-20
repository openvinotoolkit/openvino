// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, batch_to_space_op) {
    NodeBuilder::get_ops().register_factory<op::v1::BatchToSpace>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto crops_begin = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto crops_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});
    auto batch2space = make_shared<op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    NodeBuilder builder(batch2space);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
