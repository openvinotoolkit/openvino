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

TEST(attributes, softmax_op) {
    NodeBuilder::get_ops().register_factory<opset1::Softmax>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto softmax = make_shared<opset1::Softmax>(data, axis);
    NodeBuilder builder(softmax);
    auto g_softmax = ov::as_type_ptr<opset1::Softmax>(builder.create());

    EXPECT_EQ(g_softmax->get_axis(), softmax->get_axis());
}
