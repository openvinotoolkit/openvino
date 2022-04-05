// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, shapeof_op1) {
    NodeBuilder::get_ops().register_factory<op::v0::ShapeOf>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    auto shapeof = make_shared<op::v0::ShapeOf>(data);
    NodeBuilder builder(shapeof);
    auto g_shapeof = ov::as_type_ptr<op::v0::ShapeOf>(builder.create());

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, shapeof_op3) {
    NodeBuilder::get_ops().register_factory<op::v3::ShapeOf>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    auto shapeof = make_shared<op::v3::ShapeOf>(data, element::Type_t::i64);
    NodeBuilder builder(shapeof);
    auto g_shapeof = ov::as_type_ptr<op::v3::ShapeOf>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
