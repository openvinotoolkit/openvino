// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_like.hpp"

#include <gtest/gtest.h>

#include "openvino/op/concat.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, convert_like_op) {
    NodeBuilder::opset().insert<ov::op::v1::ConvertLike>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 2, 3});
    auto like = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 2, 3});

    auto convertLike = make_shared<ov::op::v1::ConvertLike>(data, like);
    NodeBuilder builder(convertLike, {data, like});
    auto g_convertLike = ov::as_type_ptr<ov::op::v0::Concat>(builder.create());

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
