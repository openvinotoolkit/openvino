// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/negative.hpp"

#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v0::Negative>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_negative, UnaryOperator, Type);

TEST(type_prop, test_negative_val_prop) {
    ov::PartialShape in_shape = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};

    auto parameter = std::make_shared<ov::op::v0::Parameter>(element::dynamic, in_shape);
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    // lower: 0, 1, 2, 3 upper: 1, 2, 3, 4
    auto constant = ov::op::v0::Constant::create(element::i64, {4}, {1, -1, 1, -1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(shape, constant);
    // lower: 0, -2, 2, -4 upper: 1, -1, 3, -3
    auto negative = std::make_shared<ov::op::v0::Negative>(mul);
    // lower: -1, 1, -3, 3 upper: 0, 2, -2, 4

    Tensor lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(negative->output(0));

    std::vector<int32_t> lower_vec = ov::op::v0::Constant(lb).cast_vector<int32_t>();
    std::vector<int32_t> lower_exp = {-1, 1, -3, 3};
    ASSERT_EQ(lower_vec, lower_exp);

    std::vector<int32_t> upper_vec = ov::op::v0::Constant(ub).cast_vector<int32_t>();
    std::vector<int32_t> upper_exp = {0, 2, -2, 4};
    ASSERT_EQ(upper_vec, upper_exp);
}
