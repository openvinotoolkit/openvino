// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, range_op) {
    NodeBuilder::get_ops().register_factory<opset4::Range>();
    auto start = make_shared<op::Parameter>(element::i64, Shape{});
    auto stop = make_shared<op::Parameter>(element::i64, Shape{});
    auto step = make_shared<op::Parameter>(element::i64, Shape{});
    auto output_type = element::f32;

    auto range = make_shared<opset4::Range>(start, stop, step, output_type);
    NodeBuilder builder(range, {start, stop, step});
    auto g_range = ov::as_type_ptr<opset4::Range>(builder.create());

    EXPECT_EQ(g_range->get_output_type(), range->get_output_type());
}
