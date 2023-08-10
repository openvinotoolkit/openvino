// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<op::v1::ReduceProd>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_et, ReduceArithmeticTest, Type);

TEST(type_prop, reduce_prod_value_propagation) {
    const auto param = std::make_shared<op::Parameter>(element::f32, PartialShape{{1, 8}, {2, 3}, 6});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceProd>(shape_of, op::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{{12, 144}}));
}
