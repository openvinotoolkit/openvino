// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, prelu_op) {
    NodeBuilder::opset().insert<ov::op::v0::PRelu>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 1, 2});
    const auto slope = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5});

    const auto prelu = make_shared<ov::op::v0::PRelu>(data, slope);
    NodeBuilder builder(prelu, {data, slope});
    EXPECT_NO_THROW(auto g_prelu = ov::as_type_ptr<ov::op::v0::PRelu>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
