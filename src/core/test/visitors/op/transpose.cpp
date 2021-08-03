// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, transpose_op)
{
    using namespace opset1;

    NodeBuilder::get_ops().register_factory<Transpose>();
    const auto data_input = make_shared<Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes_order_input = make_shared<Parameter>(element::i32, Shape{3});

    const auto op = make_shared<Transpose>(data_input, axes_order_input);

    NodeBuilder builder(op);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
