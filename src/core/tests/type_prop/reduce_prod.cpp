// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_prod.hpp"

#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "reduce_ops.hpp"

using Type = ::testing::Types<ov::op::v1::ReduceProd>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_et, ReduceArithmeticTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_dynamic, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_dynamic_zero, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_scalar, ReduceTest, Type);

TEST(type_prop, reduce_prod_value_propagation) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{1, 8}, {2, 3}, 6});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceProd>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{{12, 144}}));
}

TEST(type_prop, reduce_prod_value_propagation_dynamic) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 12, 32, 32});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceProd>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{-1}));
}

TEST(type_prop, reduce_prod_value_propagation_dynamic_zero) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 12, 0, -1});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceProd>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{0}));
}

TEST(type_prop, reduce_prod_value_propagation_scalar) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{0});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceProd>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{0}));
}
