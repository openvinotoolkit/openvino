// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, softmax_default_axis) {
    const Shape arg_shape{2, 3};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto sm = make_shared<op::v1::Softmax>(arg);
    ASSERT_EQ(sm->get_axis(), 1);
}

TEST(type_prop, softmax_out_of_bound_axis) {
    const Shape arg_shape{2, 3};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    // axis cannot be a negative number
    ASSERT_THROW(const auto unused = make_shared<op::v1::Softmax>(arg, -1), ov::NodeValidationFailure);
}

TEST(type_prop, softmax_8_default_axis) {
    const Shape arg_shape{2, 3};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto sm = make_shared<op::v8::Softmax>(arg);
    ASSERT_EQ(sm->get_axis(), 1);
}

TEST(type_prop, softmax_8_out_of_bound_negative_axis) {
    const Shape arg_shape{2, 3};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    // axis should be in range [-rank, rank - 1]
    ASSERT_THROW(const auto unused = make_shared<op::v8::Softmax>(arg, -10), ov::NodeValidationFailure);
}

TEST(type_prop, softmax_8_out_of_bound_positive_axis) {
    const Shape arg_shape{2, 3};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    // axis should be in range [-rank, rank - 1]
    ASSERT_THROW(const auto unused = make_shared<op::v8::Softmax>(arg, 10), ov::NodeValidationFailure);
}

TEST(type_prop, softmax_8_positive_axis) {
    const Shape arg_shape{1, 10};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto softmax = make_shared<op::v8::Softmax>(arg, 1);
    ASSERT_EQ(softmax->get_element_type(), element::f32);
    ASSERT_EQ(softmax->get_shape(), (Shape{1, 10}));
}

TEST(type_prop, softmax_8_negative_axis) {
    const Shape arg_shape{1, 10};
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, arg_shape);
    auto softmax = make_shared<op::v8::Softmax>(arg, -1);
    ASSERT_EQ(softmax->get_element_type(), element::f32);
    ASSERT_EQ(softmax->get_shape(), (Shape{1, 10}));
}
