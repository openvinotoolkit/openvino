// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/round.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

void static test_mode(op::v5::Round::RoundMode mode) {
    NodeBuilder::get_ops().register_factory<op::v5::Round>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{200});
    auto round = make_shared<op::v5::Round>(data, mode);
    NodeBuilder builder(round);
    auto g_round = as_type_ptr<op::v5::Round>(builder.create());

    EXPECT_EQ(g_round->get_mode(), round->get_mode());
}

TEST(attributes, round_op_enum_mode_half_to_even) {
    test_mode(op::v5::Round::RoundMode::HALF_TO_EVEN);
}

TEST(attributes, round_op_enum_mode_half_away_from_zero) {
    test_mode(op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
}
