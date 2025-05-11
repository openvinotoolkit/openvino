// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include "gtest/gtest.h"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, mish_op) {
    NodeBuilder::opset().insert<ov::op::v4::Mish>();
    const auto A = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});

    const auto mish = make_shared<ov::op::v4::Mish>(A);
    NodeBuilder builder(mish, {A});
    EXPECT_NO_THROW(auto g_mish = ov::as_type_ptr<ov::op::v4::Mish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
