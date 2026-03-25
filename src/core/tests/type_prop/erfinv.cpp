// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;

TEST(type_prop, erfinv_incorrect_type_int) {
    const auto input_type = element::i32;
    const auto input_shape = Shape{1, 3, 6};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    OV_EXPECT_THROW_HAS_SUBSTRING(std::ignore = make_shared<op::v17::ErfInv>(data),
                                  NodeValidationFailure,
                                  "Input element type must be floating-point, instead got");
}

TEST(type_prop, erfinv_f32) {
    const auto input_type = element::f32;
    const auto input_shape = Shape{1, 3, 6};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
    EXPECT_EQ(erfinv_op->get_shape(), input_shape);
}

TEST(type_prop, erfinv_f16) {
    const auto input_type = element::f16;
    const auto input_shape = Shape{2, 4};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
    EXPECT_EQ(erfinv_op->get_shape(), input_shape);
}

TEST(type_prop, erfinv_bf16) {
    const auto input_type = element::bf16;
    const auto input_shape = Shape{1};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
}

TEST(type_prop, erfinv_f64_scalar) {
    auto data = make_shared<op::v0::Parameter>(element::f64, Shape{});
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), element::f64);
    EXPECT_EQ(erfinv_op->get_shape(), Shape{});
}

// One dynamic dimension, static rank — output shape must match input.
TEST(type_prop, erfinv_dynamic_dim) {
    const auto input_shape = PartialShape{1, Dimension::dynamic(), 6};
    auto data = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), element::f32);
    EXPECT_TRUE(erfinv_op->get_output_partial_shape(0).same_scheme(input_shape));
    EXPECT_TRUE(erfinv_op->get_output_partial_shape(0).rank().is_static());
}

// Interval-bounded dimensions — bounds must be preserved in the output.
TEST(type_prop, erfinv_interval_dims) {
    const auto input_shape = PartialShape{Dimension(2, 5), Dimension(1, 4), 3};
    auto data = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), element::f32);
    EXPECT_TRUE(erfinv_op->get_output_partial_shape(0).same_scheme(input_shape));
}

// Fully-dynamic rank — output must also be rank-dynamic.
TEST(type_prop, erfinv_dynamic_rank) {
    auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), element::f32);
    EXPECT_TRUE(erfinv_op->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

// Dimension symbols must propagate unchanged through erfinv.
TEST(type_prop, erfinv_symbol_propagation) {
    auto labeled_shape = PartialShape{Dimension::dynamic(), 4};
    auto symbols = set_shape_symbols(labeled_shape);

    auto data = make_shared<op::v0::Parameter>(element::f32, labeled_shape);
    auto erfinv_op = make_shared<op::v17::ErfInv>(data);

    const auto& out_shape = erfinv_op->get_output_partial_shape(0);
    EXPECT_EQ(out_shape.rank().get_length(), 2);
    EXPECT_EQ(get_shape_symbols(out_shape), symbols);
}
