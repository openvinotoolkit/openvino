// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace testing;

class TypePropMultinomialV13Test : public TypePropOpTest<ov::op::v13::Multinomial> {};

TEST_F(TypePropMultinomialV13Test, input_probs_const_f64_num_samples_i32_convert_i32) {
    const auto probs = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto op = make_op(probs, num_samples, ov::element::i32, false, false, 0, 0);
    EXPECT_EQ(op->get_element_type(), ov::element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{2, -1}));
}

TEST_F(TypePropMultinomialV13Test, input_probs_f32_num_samples_i32_convert_i64) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    const auto op = make_op(probs, num_samples, ov::element::i64, false, false, 0, 0);
    EXPECT_EQ(op->get_element_type(), ov::element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{4, -1}));
}

TEST_F(TypePropMultinomialV13Test, input_probs_f32_num_samples_const_i32_convert_i64) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    const auto num_samples = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {10});
    const auto op = make_op(probs, num_samples, ov::element::i64, false, false, 0, 0);
    EXPECT_EQ(op->get_element_type(), ov::element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), (ov::PartialShape{4, 10}));
}

TEST_F(TypePropMultinomialV13Test, probs_incompatibile_data_type) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4, 4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = make_op(probs, num_samples, ov::element::u64, false, false, 0, 0),
                    ov::NodeValidationFailure,
                    HasSubstr("Expected floating point type as element type for the 'probs' input."));
}

TEST_F(TypePropMultinomialV13Test, num_samples_incompatibile_data_type) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{});
    OV_EXPECT_THROW(std::ignore = make_op(probs, num_samples, ov::element::u64, false, false, 0, 0),
                    ov::NodeValidationFailure,
                    HasSubstr("Expected integer type as element type for the 'num_samples' input."));
}

TEST_F(TypePropMultinomialV13Test, probs_incompatibile_rank_too_big) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4, 4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    OV_EXPECT_THROW(std::ignore = make_op(probs, num_samples, ov::element::boolean, false, false, 0, 0),
                    ov::NodeValidationFailure,
                    HasSubstr("Input probabilities must be a 2D tensor."));
}

TEST_F(TypePropMultinomialV13Test, probs_incompatibile_rank_too_small) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    OV_EXPECT_THROW(std::ignore = make_op(probs, num_samples, ov::element::boolean, false, false, 0, 0),
                    ov::NodeValidationFailure,
                    HasSubstr("Input probabilities must be a 2D tensor."));
}

TEST_F(TypePropMultinomialV13Test, num_samples_incompatibile_rank) {
    const auto probs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4});
    const auto num_samples = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 2});
    OV_EXPECT_THROW(std::ignore = make_op(probs, num_samples, ov::element::boolean, false, false, 0, 0),
                    ov::NodeValidationFailure,
                    HasSubstr("Number of samples must be a scalar or one element 1D tensor."));
}
