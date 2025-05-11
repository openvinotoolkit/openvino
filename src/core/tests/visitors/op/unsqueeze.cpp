// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, unsqueeze_op) {
    NodeBuilder::opset().insert<op::v0::Unsqueeze>();

    auto param = make_shared<op::v0::Parameter>(element::f32, Shape{4, 1, 4, 1, 8});
    auto axes = make_shared<ov::op::v0::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto op = make_shared<op::v0::Unsqueeze>(param, axes);

    NodeBuilder builder(op, {param, axes});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<op::v0::Unsqueeze>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
