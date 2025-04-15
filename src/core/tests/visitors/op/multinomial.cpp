// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include <gtest/gtest.h>

#include "openvino/op/unique.hpp"
#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, multinomial) {
    NodeBuilder::opset().insert<ov::op::v13::Multinomial>();
    const auto probs = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});

    const auto op = std::make_shared<ov::op::v13::Multinomial>(probs, num_samples, element::f32, false, true, 0, 0);
    NodeBuilder builder(op, {probs, num_samples});
    auto g_multi = ov::as_type_ptr<ov::op::v13::Multinomial>(builder.create());

    constexpr auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(op->get_with_replacement(), g_multi->get_with_replacement());
    EXPECT_EQ(op->get_global_seed(), g_multi->get_global_seed());
    EXPECT_EQ(op->get_convert_type(), g_multi->get_convert_type());
    EXPECT_EQ(op->get_log_probs(), g_multi->get_log_probs());
    EXPECT_EQ(op->get_op_seed(), g_multi->get_op_seed());
}
