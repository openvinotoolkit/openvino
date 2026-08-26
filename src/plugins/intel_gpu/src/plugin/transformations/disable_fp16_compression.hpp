// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief This transformation disables fp16 compression for RMS nodes in a specific pattern
 * to prevent precision loss.
 *
 * The targeted pattern is:
 *
 *     ...               ...
 *      |                 |
 *   Add (f32)        RMS (f32)
 * (add_m)          (rms_post_m)
 *      \              /
 *       \            /
 *         Add (f32)
 *        (add_1_m)
 *            |
 *            |
 *         RMS (f32)
 *         (rms_m)
 *
 * This pass finds the final RMS node (rms_m) in this chain and disables fp16 compression
 * for both itself and the preceding RMS node (rms_post_m). This is done to maintain
 * higher precision, as the result of the intermediate `add_1_m` operation can exceed
 * the representable range of fp16, leading to significant precision loss.
 * By keeping this pattern in fp32, numerical stability is preserved.
 */
class DisableFP16CompForGemma3RMSPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGemma3RMSPattern");
    DisableFP16CompForGemma3RMSPattern();
};

/**
 * @brief Keeps gated residual paths in FP32 when they feed normalization.
 *
 * The Multiply result in Add(residual, Multiply(gate, branch)) can exceed the
 * FP16 range even when both Multiply inputs are representable in FP16.
 */
class DisableFP16CompForGatedResidualPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGatedResidualPattern");
    DisableFP16CompForGatedResidualPattern();
};

class DisableFP16CompForDirectMultiplySinCos : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForDirectMultiplySinCos");
    DisableFP16CompForDirectMultiplySinCos();
};

class DisableFP16ComForGPTOSSROPEPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16ComForROPEPattern");
    DisableFP16ComForGPTOSSROPEPattern();
};

/**
 * @brief Keeps the F0 oscillator chain (StyleTTS2 / iSTFTNet's `l_sin_gen`)
 *        in fp32 by tagging it with `disable_fp16_compression`. The chain
 *        cumulatively integrates phase via CumSum and applies sin() at the
 *        predicted F0 (voice pitch); fp16 loses precision on the accumulated
 *        phase and produces audible noise in the F0 / first-formant band.
 *
 *  Before                                  After
 *  ------                                  -----
 *  any (FP16)  --+                         any (FP32)  --+    <- also tagged
 *                |                                       |
 *                v                                       v
 *             CumSum     (FP16)                       CumSum     (FP32)
 *                |                                       |
 *                v                                       v
 *             Multiply   (FP16)                       Multiply   (FP32)
 *                |                                       |
 *                v                                       v
 *             Transpose  (FP16)                       Transpose  (FP32)
 *                |                                       |
 *                v                                       v
 *             Multiply   (FP16)                       Multiply   (FP32)
 *                |                                       |
 *                v                                       v
 *             Interpolate(FP16)                       Interpolate(FP32)
 *                |                                       |
 *                v                                       v
 *             Transpose  (FP16)                       Transpose  (FP32)
 *                |                                       |
 *                v                                       v
 *               Sin      (FP16) -- audible noise        Sin      (FP32) -- clean
 *                |                                       |
 *                v                                       v
 *           downstream                             downstream
 *
 *  Note: only `disable_fp16_compression` rt_info is added; no graph topology
 *  change. The downstream `ConvertPrecision` pass uses the rt_info to keep
 *  the marked nodes in fp32 while the rest of the model is lowered to fp16.
 */
class DisableFP16CompCumSumSinGen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompCumSumSinGen");
    DisableFP16CompCumSumSinGen();
};

class DisableFP16ComSinGenPatternForHiFiGAN : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16ComSinGenPatternForHiFiGAN");
    DisableFP16ComSinGenPatternForHiFiGAN();
};

/**
 * @brief Runs GPU-specific matchers that mark numerically sensitive subgraphs
 *        to remain in FP32 during FP16 compression.
 */
class DisableFP16Compression : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DisableFP16Compression");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
