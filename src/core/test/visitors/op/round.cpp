// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

void static test_mode(opset5::Round::RoundMode mode)
{
    NodeBuilder::get_ops().register_factory<opset5::Round>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{200});
    auto round = make_shared<opset5::Round>(data, mode);
    NodeBuilder builder(round);
    auto g_round = as_type_ptr<opset5::Round>(builder.create());

    EXPECT_EQ(g_round->get_mode(), round->get_mode());
}

TEST(attributes, round_op_enum_mode_half_to_even)
{
    test_mode(opset5::Round::RoundMode::HALF_TO_EVEN);
}

TEST(attributes, round_op_enum_mode_half_away_from_zero)
{
    test_mode(opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
}


