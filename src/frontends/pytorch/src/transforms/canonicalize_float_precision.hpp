// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * @interface CanonicalizeFloatPrecision
 * @brief Rewrites a bf16/f16 graph into the canonical OV LLM form: f32 activations, weights kept narrow.
 *
 * torch.compile hands a model over in its own dtype, so every activation, Convert and weight is bf16
 * (or f16). The CPU LLM fusions (MLPFusion, QKVProjFusion, FullyConnectedCompressed) are written for the
 * form ovc and optimum produce, where compute precision comes from INFERENCE_PRECISION_HINT:
 *
 *     Constant(bf16) -> Convert(f32) -> MatMul(f32)
 *
 * The pass retargets narrow Converts to f32, puts large narrow Constants behind a Convert(f32) and folds
 * small ones (scalars, eps, rotary tables) to f32. Narrow Parameters get a Convert(f32) after them, except
 * the "__pa__" side-channel Parameters, whose KV-cache precision is set later. Results get a Convert back
 * to their original type, so the model's inputs and outputs keep their dtype. Converts that became
 * identities are removed.
 *
 * The new Converts are not marked here: ov::pass::MarkCompressedFloatConstants, run after this pass,
 * marks them as decompression and disables constant folding on them and their Constants.
 *
 * Enabled by the "canonical_float_precision" decoder rt_info flag (set by the torch.compile vLLM path).
 */
class CanonicalizeFloatPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::CanonicalizeFloatPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
