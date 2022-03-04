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

TEST(attributes, reshape_op) {
    NodeBuilder::get_ops().register_factory<opset1::Reshape>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    auto pattern = make_shared<op::Parameter>(element::i32, Shape{2});

    bool special_zero = true;

    auto reshape = make_shared<opset1::Reshape>(data, pattern, special_zero);
    NodeBuilder builder(reshape);
    auto g_reshape = ov::as_type_ptr<opset1::Reshape>(builder.create());

    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_reshape->get_special_zero(), reshape->get_special_zero());
}
