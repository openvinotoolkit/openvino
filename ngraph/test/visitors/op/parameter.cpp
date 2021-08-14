// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, parameter_op) {
    NodeBuilder::get_ops().register_factory<op::v0::Parameter>();
    auto parameter = std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension{1}, Dimension{4}});

    NodeBuilder builder(parameter);
    auto g_parameter = as_type_ptr<op::v0::Parameter>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_parameter->get_partial_shape(), parameter->get_partial_shape());
    EXPECT_EQ(g_parameter->get_element_type(), parameter->get_element_type());
}
