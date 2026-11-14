// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <openvino/op/gelu.hpp>

#include "../convert_common.hpp"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu::mlir {

struct ConvertGelu {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);

        auto const_tensor = [&](double val) -> ::mlir::Value {
            auto c = getConstant(builder, ov_output_element_type, val, loc);
            return linalg::FillOp::create(builder, loc, ::mlir::ValueRange{c}, ::mlir::ValueRange{empty}).getResult(0);
        };

        auto unary = [&](::mlir::Value in, linalg::ElementwiseKind kind) -> Operation* {
            return linalg::ElementwiseOp::create(builder, loc, ::mlir::ValueRange{in}, ::mlir::ValueRange{empty}, kind);
        };

        auto binary = [&](::mlir::Value lhs, ::mlir::Value rhs, linalg::ElementwiseKind kind) -> Operation* {
            return linalg::ElementwiseOp::create(builder, loc, ::mlir::ValueRange{lhs, rhs}, ::mlir::ValueRange{empty}, kind);
        };

        auto gelu_v7 = ov::as_type_ptr<ov::op::v7::Gelu>(node);
        const bool is_tanh = gelu_v7 && (gelu_v7->get_approximation_mode() == ov::op::GeluApproximationMode::TANH);

        ::mlir::Value activated;
        if (!is_tanh) {
            // ERF mode: erf(x * (1 / sqrt(2)))
            auto inv_sqrt2 = const_tensor(M_SQRT1_2);
            auto scaled = binary(input, inv_sqrt2, linalg::ElementwiseKind::mul)->getResult(0);
            activated = unary(scaled, linalg::ElementwiseKind::erf)->getResult(0);
        } else {
            // TANH mode: tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))
            auto coeff = const_tensor(0.044715);
            auto sqrt_2_over_pi = const_tensor(std::sqrt(M_2_PI));
            auto x2 = binary(input, input, linalg::ElementwiseKind::mul)->getResult(0);
            auto x3 = binary(x2, input, linalg::ElementwiseKind::mul)->getResult(0);
            auto c_x3 = binary(x3, coeff, linalg::ElementwiseKind::mul)->getResult(0);
            auto inner = binary(input, c_x3, linalg::ElementwiseKind::add)->getResult(0);
            auto arg = binary(inner, sqrt_2_over_pi, linalg::ElementwiseKind::mul)->getResult(0);
            activated = unary(arg, linalg::ElementwiseKind::tanh)->getResult(0);
        }

        auto half = const_tensor(0.5);
        auto one = const_tensor(1.0);
        auto half_x = binary(input, half, linalg::ElementwiseKind::mul)->getResult(0);
        auto plus_one = binary(activated, one, linalg::ElementwiseKind::add)->getResult(0);
        return binary(half_x, plus_one, linalg::ElementwiseKind::mul);
    }
};

}  // namespace ov::intel_gpu::mlir