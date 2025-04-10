// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;

TEST(attributes, batch_to_space_op) {
    test::NodeBuilder::opset().insert<op::v1::BatchToSpace>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 128});
    auto block_shape = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto crops_begin = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 2});
    auto crops_end = make_shared<ov::op::v0::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 0});
    auto batch2space = make_shared<op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    test::NodeBuilder builder(batch2space, {data});
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
