// Copyright (C) 2018-2022 Intel Corporation
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

TEST(attributes, reverse_op_enum_mode) {
    NodeBuilder::get_ops().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, opset1::Reverse::Mode::INDEX);
    NodeBuilder builder(reverse);
    auto g_reverse = ov::as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_op_string_mode) {
    NodeBuilder::get_ops().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    std::string mode = "index";

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, mode);
    NodeBuilder builder(reverse);
    auto g_reverse = ov::as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}
