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

TEST(attributes, atan_op)
{
    NodeBuilder::get_ops().register_factory<op::Atan>();
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1});
    const auto atan = make_shared<op::Atan>(data_node);

    const NodeBuilder builder(atan);
    const auto atan_attr_number = 0;

    EXPECT_EQ(builder.get_value_map_size(), atan_attr_number);
}
