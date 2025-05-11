// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reduce_min.hpp"

#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "reduce_ops.hpp"

using Type = ::testing::Types<ov::op::v1::ReduceMin>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_min, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_min_et, ReduceArithmeticTest, Type);

TEST(type_prop, reduce_min_value_propagation) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{{1, 8}, {2, 3}, 6});
    const auto shape_of = std::make_shared<op::v3::ShapeOf>(param);
    const auto reduce_prod =
        std::make_shared<op::v1::ReduceMin>(shape_of, ov::op::v0::Constant::create(element::i64, {1}, {0}), true);
    const auto reshape = std::make_shared<op::v1::Reshape>(param, reduce_prod, false);

    EXPECT_EQ(reshape->get_element_type(), ov::element::f32);
    EXPECT_EQ(reshape->get_output_partial_shape(0), (PartialShape{{1, 3}}));
}
