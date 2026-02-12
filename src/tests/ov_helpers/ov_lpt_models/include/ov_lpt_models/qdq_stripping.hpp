// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class QDQStrippingFunction {
public:
    // Helper: builds a weight-DQ pattern: Constant(quantized_type) -> Convert(f32) -> Subtract(zp) -> Multiply(scale)
    static ov::Output<ov::Node> build_dq_subgraph(ov::element::Type quantized_type,
                                                   const ov::Shape& shape,
                                                   float scale_value,
                                                   int zero_point = 0,
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
    static std::shared_ptr<ov::Model> getOriginalSharedDQ(const ov::PartialShape& input_shape);
    static std::shared_ptr<ov::Model> getReferenceSharedDQ(const ov::PartialShape& input_shape);

    // === NeedScalingMulMatMul pattern ===
    // Two params each multiplied by a shared DQ constant, then MatMul, then FQ→DQ→Softmax.
    // FQ y_scale = 2, so weights must be divided by 2.
    static std::shared_ptr<ov::Model> getOriginalNeedScalingMulMatMul(const ov::PartialShape& input_shape);
    static std::shared_ptr<ov::Model> getReferenceNeedScalingMulMatMul(const ov::PartialShape& input_shape);

    // === NeedScalingMatMulWithBias pattern ===
    // MatMul with DQ weights + DQ bias + Add, then FQ→DQ→MVN.
    // FQ y_scale = 4, so weights and bias must be divided by 4.
    static std::shared_ptr<ov::Model> getOriginalNeedScalingMatMulWithBias(const ov::PartialShape& input_shape);
    static std::shared_ptr<ov::Model> getReferenceNeedScalingMatMulWithBias(const ov::PartialShape& input_shape);

    // === NeedScalingResidualBlock pattern ===
    // Conv→bias→FQ→DQ→FQ(fwd)→residual_blocks(MVN→Conv→bias→FQ(branch)→Add)→MVN
    // First FQ y_scale=10 → stripped + scale propagation.
    // Forward-path FQ and branch FQs adjusted then stripped.
    static std::shared_ptr<ov::Model> getOriginalNeedScalingResidualBlock(const ov::PartialShape& input_shape);
    static std::shared_ptr<ov::Model> getReferenceNeedScalingResidualBlock(const ov::PartialShape& input_shape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
