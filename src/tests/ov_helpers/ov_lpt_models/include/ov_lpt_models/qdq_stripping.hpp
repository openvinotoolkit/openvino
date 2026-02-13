// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace builder {
namespace subgraph {

// Encapsulates FQ input/output range and zero point for a given quantization precision.
// Used to build FQ+DQ pairs where output_low/output_high match the precision limits
// (e.g., [-32768, 32767] for i16, [0, 65535] for u16) and input_low/input_high define
// the dequantized value range.
class QuantizationParams {
public:
    ov::Output<ov::Node> build_fq(const ov::Output<ov::Node>& input) const;
    ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input, const ov::element::Type& quantization_precision) const;

    float i_l;
    float i_h;
    float o_l;
    float o_h;
    int zero_point;
};

class QDQStrippingFunction {
public:
    // Helper: builds a weight-DQ pattern: Constant(quantized_type) -> Convert(f32) -> Subtract(zp) -> Multiply(scale)
    // When seed is provided, generates random constant values using make_constant.
    // When constant_values vector is provided, uses those values directly.
    // Otherwise uses single constant_value (default 1.f).
    static ov::Output<ov::Node> build_dq_subgraph(ov::element::Type quantized_type,
                                                   const ov::Shape& shape,
                                                   float scale_value,
                                                   int zero_point = 0,
                                                   std::optional<size_t> seed = std::nullopt,
                                                   std::optional<std::vector<int>> constant_values = std::nullopt,
                                                   float constant_value = 1.f);

    // Helper: builds Conv bias pattern using ShapeOf-based reshape
    static ov::Output<ov::Node> add_bias(const ov::Output<ov::Node>& conv,
                                         const ov::Output<ov::Node>& bias);

    // Helper: creates an FQ with 65536 levels and scalar range constants
    static ov::Output<ov::Node> build_fq(const ov::Output<ov::Node>& input,
                                         float input_low, float input_high,
                                         float output_low, float output_high,
                                         size_t levels = 65536);

    // Helper: creates a DQ (Convert -> Subtract -> Multiply) from quantization precision
    static ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input,
                                         const ov::element::Type& quantization_precision,
                                         float input_low, float input_high,
                                         float output_low, float output_high,
                                         int zero_point = 0);

    // === SharedDQ pattern ===
    // Two Conv branches sharing a quantized input (FQ → Convert → Convert → DQ → Conv + FQ → DQ)
    // FQs have y_scale < 1, so they are just stripped without scale propagation.
    static std::shared_ptr<ov::Model> build_shared_dq_pattern_ref(const ov::PartialShape& input_shape,
                                                                   const ov::element::Type& quantization_precision,
                                                                   bool need_weights_adjustment = true);

    // === NeedScalingMulMatMul pattern ===
    // Two params each multiplied by a shared DQ constant, then MatMul, then FQ→DQ→Softmax.
    // When need_weights_adjustment=true: FQ y_scale=2, weights divided by y_scale*ratio.
    // When need_weights_adjustment=false: FQ stripped, weights unchanged.
    static std::shared_ptr<ov::Model> build_need_scaling_mul_matmul_pattern_ref(const ov::PartialShape& input_shape,
                                                                                const ov::element::Type& quantization_precision,
                                                                                bool need_weights_adjustment = true);

    // === NeedScalingMatMulWithBias pattern ===
    // MatMul with DQ weights + DQ bias + Add, then FQ→DQ→MVN.
    // When need_weights_adjustment=true: FQ y_scale=4, weights/bias divided by y_scale*ratio.
    // When need_weights_adjustment=false: FQ stripped, weights/bias unchanged.
    static std::shared_ptr<ov::Model> build_need_scaling_matmul_with_bias_pattern_ref(const ov::PartialShape& input_shape,
                                                                                      const ov::element::Type& quantization_precision,
                                                                                      bool need_weights_adjustment = true);

    // === NeedScalingResidualBlock pattern ===
    // Conv→bias→FQ→DQ→FQ(fwd)→residual_blocks(MVN→Conv→bias→FQ(branch)→Add)→MVN
    // When need_weights_adjustment=true: first FQ y_scale=10, scale propagation adjusts weights/FQs.
    // When need_weights_adjustment=false: all FQs stripped, weights unchanged.
    static std::shared_ptr<ov::Model> build_need_scaling_residual_block_pattern_ref(const ov::PartialShape& input_shape,
                                                                                    const ov::element::Type& quantization_precision,
                                                                                    bool need_weights_adjustment = true);

    // === GPU accuracy test model builders ===
    // These build full models with per-precision QuantizationParams (i16/u16) for
    // ConvertQuantizeDequantize fusion compatibility. Used by both LPT unit tests and
    // GPU functional tests.
    static std::shared_ptr<ov::Model> build_shared_dq_pattern(const ov::PartialShape& input_shape,
                                                               const ov::element::Type& quantization_precision);

    static std::shared_ptr<ov::Model> build_need_scaling_mul_matmul_pattern(const ov::PartialShape& input_shape,
                                                                            const ov::element::Type& quantization_precision);

    static std::shared_ptr<ov::Model> build_need_scaling_matmul_with_bias_pattern(const ov::PartialShape& input_shape,
                                                                                   const ov::element::Type& quantization_precision);

    static std::shared_ptr<ov::Model> build_need_scaling_residual_block_pattern(const ov::PartialShape& input_shape,
                                                                                const ov::element::Type& quantization_precision);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
