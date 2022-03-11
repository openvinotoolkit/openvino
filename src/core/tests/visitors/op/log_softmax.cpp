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

TEST(attributes, logsoftmax_op) {
    NodeBuilder::get_ops().register_factory<opset5::LogSoftmax>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 2, 3});

    int64_t axis = 2;

    const auto logsoftmax = make_shared<opset5::LogSoftmax>(data, axis);
    NodeBuilder builder(logsoftmax);
    auto g_logsoftmax = ov::as_type_ptr<opset5::LogSoftmax>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_logsoftmax->get_axis(), logsoftmax->get_axis());
}
