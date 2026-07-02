// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"

using namespace ov;
using namespace testing;

namespace v0 = ov::op::v0;
using ov::op::internal::FullyConnectedCompressed;
using ov::op::internal::FullyConnectedQuantized;
using ov::op::internal::FullyConnectedQuantizedLegacy;

namespace {

// X: [M, K] = [3, 4], W: [N, K] = [5, 4] (MatMul uses transpose_b), output: [M, N] = [3, 5].
std::shared_ptr<v0::Parameter> data() {
    return std::make_shared<v0::Parameter>(element::f32, PartialShape{3, 4});
}
std::shared_ptr<v0::Parameter> weights() {
    return std::make_shared<v0::Parameter>(element::f32, PartialShape{5, 4});
}
std::shared_ptr<Node> num_const(element::Type et, const Shape& s = Shape{5, 1}) {
    return v0::Constant::create(et, s, {0});
}
// Absent optional inputs (bias / zero-points) are passed as empty, dynamic-typed constants.
std::shared_ptr<Node> empty_dyn() {
    return std::make_shared<v0::Constant>(element::dynamic, Shape{0});
}

}  // namespace

// ---------------------------------------------------------------------------
// FullyConnectedCompressed (5 inputs)
// ---------------------------------------------------------------------------

TEST(type_prop_fc_compressed, valid_output_type_and_shape) {
    auto op = std::make_shared<FullyConnectedCompressed>(data(),
                                                         weights(),
                                                         empty_dyn(),
                                                         num_const(element::f32),
                                                         num_const(element::u8));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 5}));
}

TEST(type_prop_fc_compressed, absent_zero_points_accepted) {
    // 4-argument constructor fills weight_zero_points with an empty dynamic constant.
    auto op = std::make_shared<FullyConnectedCompressed>(data(), weights(), empty_dyn(), num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_compressed, integral_scales_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedCompressed>(data(),
                                                               weights(),
                                                               empty_dyn(),
                                                               num_const(element::i32),
                                                               num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("weight_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_compressed, float_zero_points_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedCompressed>(data(),
                                                               weights(),
                                                               empty_dyn(),
                                                               num_const(element::f32),
                                                               num_const(element::f32)),
                    NodeValidationFailure,
                    HasSubstr("weight_zero_points (input 4) must have an integral"));
}

// ---------------------------------------------------------------------------
// FullyConnectedQuantized (9 inputs)
// ---------------------------------------------------------------------------

TEST(type_prop_fc_quantized, valid_output_type_and_shape) {
    auto op = std::make_shared<FullyConnectedQuantized>(data(),
                                                        weights(),
                                                        empty_dyn(),
                                                        num_const(element::f32),  // weight_scales
                                                        num_const(element::u8),   // weight_zero_points
                                                        num_const(element::f32),  // input_scales
                                                        num_const(element::u8),   // input_zero_points
                                                        num_const(element::f32),  // output_scales
                                                        num_const(element::u8));  // output_zero_points
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 5}));
}

TEST(type_prop_fc_quantized, integral_input_scales_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedQuantized>(data(),
                                                              weights(),
                                                              empty_dyn(),
                                                              num_const(element::f32),
                                                              num_const(element::u8),
                                                              num_const(element::i32),
                                                              num_const(element::u8),
                                                              num_const(element::f32),
                                                              num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("input_scales (input 5) must have a floating-point"));
}

TEST(type_prop_fc_quantized, float_output_zero_points_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedQuantized>(data(),
                                                              weights(),
                                                              empty_dyn(),
                                                              num_const(element::f32),
                                                              num_const(element::u8),
                                                              num_const(element::f32),
                                                              num_const(element::u8),
                                                              num_const(element::f32),
                                                              num_const(element::f32)),
                    NodeValidationFailure,
                    HasSubstr("output_zero_points (input 8) must have an integral"));
}

// ---------------------------------------------------------------------------
// FullyConnectedQuantizedLegacy (5 inputs)
// ---------------------------------------------------------------------------

TEST(type_prop_fc_quantized_legacy, integral_zero_points_accepted) {
    auto op = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                              weights(),
                                                              empty_dyn(),
                                                              num_const(element::f32),
                                                              num_const(element::u8));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 5}));
}

TEST(type_prop_fc_quantized_legacy, real_zero_points_accepted) {
    // Legacy dequant may subtract a real zero-point, so a floating-point type is allowed here.
    auto op = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                              weights(),
                                                              empty_dyn(),
                                                              num_const(element::f32),
                                                              num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_quantized_legacy, integral_scales_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                                    weights(),
                                                                    empty_dyn(),
                                                                    num_const(element::i32),
                                                                    num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("deq_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_quantized_legacy, boolean_zero_points_throw) {
    OV_EXPECT_THROW(std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                                    weights(),
                                                                    empty_dyn(),
                                                                    num_const(element::f32),
                                                                    num_const(element::boolean)),
                    NodeValidationFailure,
                    HasSubstr("deq_zero_points (input 4) must have a numeric"));
}
