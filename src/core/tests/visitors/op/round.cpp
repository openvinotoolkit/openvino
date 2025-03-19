// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

void static test_mode(ov::op::v5::Round::RoundMode mode) {
    NodeBuilder::opset().insert<ov::op::v5::Round>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{200});
    auto round = make_shared<ov::op::v5::Round>(data, mode);
    NodeBuilder builder(round, {data});
    auto g_round = ov::as_type_ptr<ov::op::v5::Round>(builder.create());

    EXPECT_EQ(g_round->get_mode(), round->get_mode());
}

TEST(attributes, round_op_enum_mode_half_to_even) {
    test_mode(ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
}

TEST(attributes, round_op_enum_mode_half_away_from_zero) {
    test_mode(ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
}
