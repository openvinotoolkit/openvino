// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_elements_op) {
    NodeBuilder::get_ops().register_factory<opset6::GatherElements>();
    auto arg1 = make_shared<opset1::Parameter>(element::i32, PartialShape{3});
    auto arg2 = make_shared<opset1::Parameter>(element::i32, PartialShape{7});
    int64_t axis = 0;

    auto gather_el = make_shared<opset6::GatherElements>(arg1, arg2, axis);
    NodeBuilder builder(gather_el);
    auto g_gather_el = ov::as_type_ptr<opset6::GatherElements>(builder.create());

    EXPECT_EQ(g_gather_el->get_axis(), gather_el->get_axis());
}
