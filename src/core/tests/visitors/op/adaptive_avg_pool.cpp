// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_avg_pool.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

TEST(attributes, adaptive_avg_pool_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v8::AdaptiveAvgPool>();
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 5, 4});
    const auto out_shape = ov::op::v0::Constant::create<int64_t>(ov::element::i64, ov::Shape{2}, {4, 3});

    const auto adaptive_pool = std::make_shared<ov::op::v8::AdaptiveAvgPool>(A, out_shape);
    ov::test::NodeBuilder builder(adaptive_pool, {A, out_shape});

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
