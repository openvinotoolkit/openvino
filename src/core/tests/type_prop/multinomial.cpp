// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

TEST(type_prop, multinomial_i64_i32) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    auto multinomial_func =
        std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::i32, false, false, 0, 0);
    EXPECT_EQ(multinomial_func->get_element_type(), ov::element::i32);
    EXPECT_EQ(multinomial_func->get_shape(), (ov::Shape{4}));
}

TEST(type_prop, multinomial_f32_i64) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    auto multinomial_func =
        std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::i64, false, false, 0, 0);
    EXPECT_EQ(multinomial_func->get_element_type(), ov::element::i64);
    EXPECT_EQ(multinomial_func->get_shape(), (ov::Shape{4}));
}

TEST(type_prop, multinomial_f16_i64) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    auto multinomial_func =
        std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::i64, false, false, 0, 0);
    EXPECT_EQ(multinomial_func->get_element_type(), ov::element::i64);
    EXPECT_EQ(multinomial_func->get_shape(), (ov::Shape{4}));
}

TEST(type_prop, multinomial_incompatibile_convert_u64) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    ASSERT_THROW(const auto unused =
                     std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::u64, false, false, 0, 0),
                 ov::NodeValidationFailure);
}

TEST(type_prop, multinomial_incompatibile_convert_u32) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    ASSERT_THROW(const auto unused =
                     std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::u32, false, false, 0, 0),
                 ov::NodeValidationFailure);
}

TEST(type_prop, multinomial_incompatibile_convert_u16) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    ASSERT_THROW(const auto unused =
                     std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::u16, false, false, 0, 0),
                 ov::NodeValidationFailure);
}

TEST(type_prop, multinomial_incompatibile_convert_bool) {
    auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    auto num_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4});
    ASSERT_THROW(
        const auto unused =
            std::make_shared<ov::op::v13::Multinomial>(probs, num_shape, ov::element::boolean, false, false, 0, 0),
        ov::NodeValidationFailure);
}
