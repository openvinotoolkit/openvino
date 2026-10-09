// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/fully_connected_quantized.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"

namespace ov {
namespace test {

using ov::op::internal::FullyConnectedCompressed;
using ov::op::internal::FullyConnectedQuantized;
using ov::op::internal::FullyConnectedQuantizedLegacy;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using testing::HasSubstr;

namespace {

// X: [M, K] = [3, 4], W: [N, K] = [5, 4] (MatMul uses transpose_b), output: [M, N] = [3, 5].
std::shared_ptr<Parameter> data() {
    return std::make_shared<Parameter>(element::f32, PartialShape{3, 4});
}
std::shared_ptr<Parameter> weights() {
    return std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
}
std::shared_ptr<Node> num_const(element::Type et, const Shape& s = Shape{5, 1}) {
    return Constant::create(et, s, {0});
}
// String is the only concrete element type that is not numeric, so it is the only one the
// zero-points check rejects.
std::shared_ptr<Node> string_const(const Shape& s = Shape{5, 1}) {
    return std::make_shared<Constant>(element::string, s, std::vector<std::string>(shape_size(s), "0"));
}
// Absent optional inputs (bias / zero-points) are passed as empty, dynamic-typed constants.
std::shared_ptr<Node> empty_dyn() {
    return std::make_shared<Constant>(element::dynamic, Shape{0});
}
// A valid FullyConnectedQuantized input list: X, W, bias, then (scales, zero-points) for weights,
// input and output.
OutputVector valid_quantized_args() {
    return {data(),
            weights(),
            empty_dyn(),
            num_const(element::f32),
            num_const(element::u8),
            num_const(element::f32),
            num_const(element::u8),
            num_const(element::f32),
            num_const(element::u8)};
}
std::shared_ptr<Node> make_quantized(const OutputVector& a) {
    return std::make_shared<FullyConnectedQuantized>(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
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
    // The 4-argument delegating constructor calls the main constructor with an element::dynamic
    // placeholder at the weight_zero_points position, so it must validate cleanly.
    auto op = std::make_shared<FullyConnectedCompressed>(data(), weights(), empty_dyn(), num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_compressed, integral_scales_throw) {
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedCompressed>(data(),
                                                                             weights(),
                                                                             empty_dyn(),
                                                                             num_const(element::i32),
                                                                             num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("weight_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_compressed, real_zero_points_accepted) {
    // Weight compression may carry a real (e.g. f32) zero-point that is subtracted before scaling,
    // as real 4-bit-compressed MatMul models emit. A floating-point zero-points type must be accepted.
    auto op = std::make_shared<FullyConnectedCompressed>(data(),
                                                         weights(),
                                                         empty_dyn(),
                                                         num_const(element::f32),
                                                         num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_compressed, boolean_zero_points_accepted) {
    auto op = std::make_shared<FullyConnectedCompressed>(data(),
                                                         weights(),
                                                         empty_dyn(),
                                                         num_const(element::f32),
                                                         num_const(element::boolean));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_compressed, dynamic_scales_throw) {
    // A dynamic type is how an absent optional input is encoded, so it cannot stand in for the
    // mandatory scales.
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedCompressed>(data(),
                                                                             weights(),
                                                                             empty_dyn(),
                                                                             empty_dyn(),
                                                                             num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("weight_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_compressed, string_zero_points_throw) {
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedCompressed>(data(),
                                                                             weights(),
                                                                             empty_dyn(),
                                                                             num_const(element::f32),
                                                                             string_const()),
                    NodeValidationFailure,
                    HasSubstr("weight_zero_points (input 4) must have a numeric"));
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

TEST(type_prop_fc_quantized, float_output_zero_points_accepted) {
    // Zero-points may be real (subtracted before scaling), so a floating-point output_zero_points
    // type must be accepted for the quantized op as well.
    auto op = std::make_shared<FullyConnectedQuantized>(data(),
                                                        weights(),
                                                        empty_dyn(),
                                                        num_const(element::f32),
                                                        num_const(element::u8),
                                                        num_const(element::f32),
                                                        num_const(element::u8),
                                                        num_const(element::f32),
                                                        num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_quantized, boolean_zero_points_accepted) {
    auto args = valid_quantized_args();
    args[4] = args[6] = args[8] = num_const(element::boolean);
    auto op = make_quantized(args);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_quantized, each_typed_input_is_checked_at_its_own_index) {
    // Scales reject non-real and dynamic types; zero-points reject string. Every one of the six
    // typed inputs is exercised under its own index and name, so a mis-numbered or mis-named check
    // cannot hide behind the inputs that happen to have a dedicated test.
    struct Case {
        size_t idx;
        const char* name;
        std::shared_ptr<Node> bad;
    };
    const std::vector<Case> cases{
        {3, "weight_scales", num_const(element::i32)},
        {3, "weight_scales", empty_dyn()},
        {4, "weight_zero_points", string_const()},
        {5, "input_scales", num_const(element::i32)},
        {5, "input_scales", empty_dyn()},
        {6, "input_zero_points", string_const()},
        {7, "output_scales", num_const(element::i32)},
        {7, "output_scales", empty_dyn()},
        {8, "output_zero_points", string_const()},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(std::string(c.name) + " = " + c.bad->get_element_type().get_type_name());
        auto args = valid_quantized_args();
        args[c.idx] = c.bad;
        OV_EXPECT_THROW(std::ignore = make_quantized(args),
                        NodeValidationFailure,
                        HasSubstr(std::string(c.name) + " (input " + std::to_string(c.idx) + ") must have a"));
    }
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

TEST(type_prop_fc_quantized_legacy, absent_zero_points_accepted) {
    // The 4-argument delegating constructor calls the main constructor with an element::dynamic
    // placeholder at the deq_zero_points position, so it must validate cleanly.
    auto op = std::make_shared<FullyConnectedQuantizedLegacy>(data(), weights(), empty_dyn(), num_const(element::f32));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
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
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                                                  weights(),
                                                                                  empty_dyn(),
                                                                                  num_const(element::i32),
                                                                                  num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("deq_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_quantized_legacy, boolean_zero_points_accepted) {
    auto op = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                              weights(),
                                                              empty_dyn(),
                                                              num_const(element::f32),
                                                              num_const(element::boolean));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop_fc_quantized_legacy, dynamic_scales_throw) {
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                                                  weights(),
                                                                                  empty_dyn(),
                                                                                  empty_dyn(),
                                                                                  num_const(element::u8)),
                    NodeValidationFailure,
                    HasSubstr("deq_scales (input 3) must have a floating-point"));
}

TEST(type_prop_fc_quantized_legacy, string_zero_points_throw) {
    OV_EXPECT_THROW(std::ignore = std::make_shared<FullyConnectedQuantizedLegacy>(data(),
                                                                                  weights(),
                                                                                  empty_dyn(),
                                                                                  num_const(element::f32),
                                                                                  string_const()),
                    NodeValidationFailure,
                    HasSubstr("deq_zero_points (input 4) must have a numeric"));
}

}  // namespace test
}  // namespace ov
