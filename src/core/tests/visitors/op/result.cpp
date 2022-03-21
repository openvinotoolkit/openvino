// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset9.hpp"
#include "unary_ops.hpp"

using ngraph::test::NodeBuilder;

using Types = ::testing::Types<UnaryOperatorType<ngraph::op::v0::Result, ngraph::element::f32>,
                               UnaryOperatorType<ngraph::op::v0::Result, ngraph::element::f16>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_without_attribute, UnaryOperatorVisitor, Types, UnaryOperatorTypeName);

TEST(attributes, result_v9_op) {
    NodeBuilder::get_ops().register_factory<ov::opset9::Parameter>();
    auto parameter = std::make_shared<ov::opset9::Parameter>(ov::element::f32,
                                                             ov::PartialShape{ov::Dimension{1}, ov::Dimension{4}},
                                                             "input_name");
    auto result = std::make_shared<ov::opset9::Result>(parameter, "tensor_name");

    NodeBuilder builder(result);
    auto g_result = ov::as_type_ptr<ov::opset9::Result>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_result->get_tensor_name(), result->get_tensor_name());
}
