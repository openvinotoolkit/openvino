// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan2.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

using namespace ov;
using namespace testing;

// ---- shape inference ----

TEST(type_prop, atan2_same_shape) {
    auto y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3}));
}

TEST(type_prop, atan2_numpy_broadcast) {
    auto y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), (Shape{2, 3, 4}));
}

TEST(type_prop, atan2_scalar_broadcast) {
    auto y = std::make_shared<op::v0::Parameter>(element::f64, Shape{3, 4});
    auto x = std::make_shared<op::v0::Parameter>(element::f64, Shape{1});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::f64);
    EXPECT_EQ(op->get_shape(), (Shape{3, 4}));
}

TEST(type_prop, atan2_f16) {
    auto y = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 2});
    auto x = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 2});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::f16);
    EXPECT_EQ(op->get_shape(), (Shape{2, 2}));
}

TEST(type_prop, atan2_bf16) {
    auto y = std::make_shared<op::v0::Parameter>(element::bf16, Shape{4});
    auto x = std::make_shared<op::v0::Parameter>(element::bf16, Shape{4});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::bf16);
}

TEST(type_prop, atan2_dynamic_shape) {
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, -1});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 4}));
}

TEST(type_prop, atan2_default_autob_is_numpy) {
    auto y = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2});
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 2});
    auto op = std::make_shared<op::v17::Atan2>(y, x);

    EXPECT_EQ(op->get_autob(), op::AutoBroadcastType::NUMPY);
}

// ---- type validation ----

TEST(type_prop, atan2_integer_input_throws) {
    auto y = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 2});
    auto x = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v17::Atan2>(y, x),
                    ov::NodeValidationFailure,
                    HasSubstr("Atan2 inputs must be floating-point type"));
}
