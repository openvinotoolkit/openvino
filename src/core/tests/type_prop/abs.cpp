// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v0::Abs>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_abs, UnaryOperator, Type);

TEST(type_prop, test_abs_non_negative_val_prop) {
    ov::PartialShape in_shape = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    set_shape_symbols(in_shape);

    auto parameter = std::make_shared<ov::op::v0::Parameter>(element::dynamic, in_shape);
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    auto abs = std::make_shared<ov::op::v0::Abs>(shape);

    ov::PartialShape output_value_as_shape;
    ov::util::evaluate_as_partial_shape(abs->output(0), output_value_as_shape);

    ASSERT_EQ(output_value_as_shape, in_shape);
    ASSERT_EQ(get_shape_symbols(output_value_as_shape), get_shape_symbols(in_shape));
}

TEST(type_prop, test_abs_negative_val_prop) {
    ov::PartialShape in_shape = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    set_shape_symbols(in_shape);

    auto parameter = std::make_shared<ov::op::v0::Parameter>(element::dynamic, in_shape);
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    // lower: 0, 1, 2, 3 upper: 1, 2, 3, 4
    auto neg = std::make_shared<ov::op::v0::Negative>(shape);
    // lower: -1, -2, -3, -4 upper: 0, -1, -2, -3
    auto abs = std::make_shared<ov::op::v0::Abs>(neg);
    // no estimation
    ov::PartialShape output_value_as_shape;
    ASSERT_EQ(ov::util::evaluate_as_partial_shape(abs->output(0), output_value_as_shape), false);
}