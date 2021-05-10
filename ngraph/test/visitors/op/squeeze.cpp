// Copyright (C) 2018-2021 Intel Corporation
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
using ngraph::test::ValueMap;

TEST(attributes, squeeze_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Squeeze>();
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto squeeze = make_shared<op::Squeeze>(data_node);

    NodeBuilder builder(squeeze);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
