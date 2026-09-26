// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_quantized_legacy.hpp"

namespace {

// X: [M, K] = [3, 4], W: [N, K] = [5, 4] (FullyConnected uses transpose_b), output: [M, N] = [3, 5].
std::shared_ptr<ov::op::v0::Parameter> activations() {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{3, 4});
}
std::shared_ptr<ov::op::v0::Constant> weights() {
    return ov::op::v0::Constant::create(ov::element::i8, ov::Shape{5, 4}, {1});
}
// Absent optional inputs (bias / zero-points) are passed as empty, dynamic-typed constants.
std::shared_ptr<ov::op::v0::Constant> empty_dyn() {
    return std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertFCToFCQuantizedLegacy_RealScales) {
    {
        auto x = activations();
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(x, weights(), empty_dyn(), ov::element::f32);
        auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 5}, {0.5f});
        auto mul = std::make_shared<ov::op::v1::Multiply>(fc, scales);
        model = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{x});
        manager.register_pass<ov::pass::ConvertFCToFCQuantizedLegacy>();
    }
    {
        auto x = activations();
        auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 5}, {0.5f});
        auto fc = std::make_shared<ov::op::internal::FullyConnectedQuantizedLegacy>(x,
                                                                                    weights(),
                                                                                    empty_dyn(),
                                                                                    scales,
                                                                                    empty_dyn(),
                                                                                    ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{x});
    }
}

// FullyConnectedQuantizedLegacy requires real-typed dequantization scales, so a Multiply by a
// non-real constant is not a dequantization: the pass must leave it alone rather than build an op
// that fails validation mid-transformation and aborts compilation.
TEST_F(TransformationTestsF, ConvertFCToFCQuantizedLegacy_IntegralScalesNotConverted) {
    auto x = activations();
    // With no output_type override the FullyConnected emits its activation type (u8), so a plain
    // Multiply after it takes a u8 constant.
    auto fc = std::make_shared<ov::op::internal::FullyConnected>(x, weights(), empty_dyn());
    auto scales = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1, 5}, {2});
    auto mul = std::make_shared<ov::op::v1::Multiply>(fc, scales);
    model = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{x});
    manager.register_pass<ov::pass::ConvertFCToFCQuantizedLegacy>();
}
