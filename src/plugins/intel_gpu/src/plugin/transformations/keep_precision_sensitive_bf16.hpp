// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps the precision sensitive subgraphs in f32 when the model is compressed to bf16.
 *
 * ov::pass::ConvertPrecision runs ov::pass::MarkSugraphsToKeepInMixedPrecision and
 * ov::pass::AlignMixedFP32FP16Types itself, but the marks left by the passes that run before it
 * (LPT, ov::pass::KeepDequantizationPrecision, ...) always use element::f16 as the key, so under a
 * bf16 compression target those subgraphs are not protected. This pass mirrors the f16 marks onto
 * the bf16 target before the conversion takes place.
 *
 * On top of that it marks the attention QK-normalization: the RMS normalization applied to the
 * query and key tensors before they enter ScaledDotProductAttention. Only the RMS node itself is
 * kept in f32 -- MarkSugraphsToKeepInMixedPrecision does not propagate the mark through the
 * Transpose that follows it (Transpose is not in its propagate_through_ops list), so the
 * normalized Q/K values are still rounded to bf16 before they reach the SDPA MatMul. The accuracy
 * this buys comes from computing the normalization's own reduction (mean of squares over the head
 * dimension, plus the division) in f32: bf16's 7-bit mantissa is not enough to keep that reduction
 * accurate, and every downstream op inherits its error. f16 keeps the same reduction in f32
 * through the div-with-eps pattern of MarkSugraphsToKeepInMixedPrecision, which does not trigger
 * for bf16 because every practical normalization epsilon is exactly representable in bf16.
 *
 * This pass only recognizes the fused ov::op::internal::RMS form of the normalization (the
 * decomposed x / sqrt(mean(x^2) + eps) * gamma chain is not matched). RMSFusion runs earlier in
 * the same pipeline but is skipped per-node for large last dimensions on some devices
 * (transformations_pipeline.cpp, RMSFusionMatcher callback); if that happens, or if a model's
 * QK-normalization uses a different primitive (LayerNorm, MVN, NormalizeL2), this pass silently
 * marks nothing and the bf16 accuracy regression it exists to fix returns. Build with
 * ENABLE_DEBUG_CAPS and a verbose level >= LOG to get a diagnostic when that happens.
 *
 * Must run before the f32 -> bf16 ov::pass::ConvertPrecision, while the model is still all f32,
 * and after ov::pass::RMSFusion so that the QK-normalization is in its fused form.
 */
class KeepPrecisionSensitiveSubgraphsForBF16 : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::intel_gpu::KeepPrecisionSensitiveSubgraphsForBF16");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
