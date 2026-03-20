// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace ov;

TEST(type_prop, erfinv_incorrect_type_int) {
    const auto input_type = element::i32;
    const auto input_shape = Shape{1, 3, 6};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    try {
        auto erfinv_op = make_shared<op::v16::ErfInv>(data);
        FAIL() << "Expected validation error for integer input type.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element type must be floating-point, instead got"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason.";
    }
}

TEST(type_prop, erfinv_f32) {
    const auto input_type = element::f32;
    const auto input_shape = Shape{1, 3, 6};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v16::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
    EXPECT_EQ(erfinv_op->get_shape(), input_shape);
}

TEST(type_prop, erfinv_f16) {
    const auto input_type = element::f16;
    const auto input_shape = Shape{2, 4};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v16::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
    EXPECT_EQ(erfinv_op->get_shape(), input_shape);
}

TEST(type_prop, erfinv_bf16) {
    const auto input_type = element::bf16;
    const auto input_shape = Shape{1};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v16::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
}

TEST(type_prop, erfinv_f32_partial) {
    const auto input_type = element::f32;
    const auto input_shape = PartialShape{1, Dimension::dynamic(), 6};
    auto data = make_shared<op::v0::Parameter>(input_type, input_shape);
    auto erfinv_op = make_shared<op::v16::ErfInv>(data);
    EXPECT_EQ(erfinv_op->get_element_type(), input_type);
    ASSERT_TRUE(erfinv_op->get_output_partial_shape(0).same_scheme(input_shape));
    ASSERT_TRUE(erfinv_op->get_output_partial_shape(0).rank().is_static());

    auto erfinv_dyn = make_shared<op::v16::ErfInv>(
        make_shared<op::v0::Parameter>(input_type, PartialShape::dynamic()));
    ASSERT_TRUE(erfinv_dyn->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}
