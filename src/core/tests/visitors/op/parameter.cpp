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

using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, parameter_op) {
    NodeBuilder::get_ops().register_factory<opset1::Parameter>();
    auto parameter = std::make_shared<op::Parameter>(element::f32, PartialShape{Dimension{1}, Dimension{4}});

    NodeBuilder builder(parameter);
    auto g_parameter = ov::as_type_ptr<opset1::Parameter>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_parameter->get_partial_shape(), parameter->get_partial_shape());
    EXPECT_EQ(g_parameter->get_element_type(), parameter->get_element_type());
}
