// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset9.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, parameter_v0_op) {
    NodeBuilder::get_ops().register_factory<ov::opset1::Parameter>();
    auto parameter = std::make_shared<ov::opset1::Parameter>(element::f32, PartialShape{Dimension{1}, Dimension{4}});

    NodeBuilder builder(parameter);
    auto g_parameter = ov::as_type_ptr<ov::opset1::Parameter>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_parameter->get_partial_shape(), parameter->get_partial_shape());
    EXPECT_EQ(g_parameter->get_element_type(), parameter->get_element_type());
}

TEST(attributes, parameter_v9_op) {
    NodeBuilder::get_ops().register_factory<ov::opset9::Parameter>();
    auto parameter =
        std::make_shared<ov::opset9::Parameter>(element::f32, PartialShape{Dimension{1}, Dimension{4}}, "tensor_name");

    NodeBuilder builder(parameter);
    auto g_parameter = ov::as_type_ptr<ov::opset9::Parameter>(builder.create());

    const auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_parameter->get_partial_shape(), parameter->get_partial_shape());
    EXPECT_EQ(g_parameter->get_element_type(), parameter->get_element_type());
    EXPECT_EQ(g_parameter->get_tensor_name(), parameter->get_tensor_name());
}
