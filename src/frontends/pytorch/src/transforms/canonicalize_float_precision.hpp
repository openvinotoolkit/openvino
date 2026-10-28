// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::frontend::pytorch::pass {

/**
 * @interface CanonicalizeFloatPrecision
 * @brief Rewrites a bf16/f16 model into the standard OpenVINO form: f32 activations, compressed weights.
 *
 * torch.compile hands a model over in its own dtype, so every activation and weight is bf16 (or f16).
 * The standard OpenVINO form keeps the weights compressed behind a decompression Convert and computes in
 * f32, leaving the execution precision to the INFERENCE_PRECISION_HINT property:
 *
 *     Constant(bf16) -> Convert(f32) -> MatMul(f32)
 *
 * Large narrow Constants are put behind a Convert(f32) and kept narrow; small ones (scalars, eps, rotary
 * tables) are folded to f32. The rest of the graph is retyped by ov::pass::ConvertPrecision, which also
 * handles ops whose output type is an attribute (Range, Eye, RandomUniform, ...). The model's inputs and
 * outputs keep their type, as do the "__pa__" side-channel Parameters, whose KV-cache precision is set
 * later. Models with subgraph ops are left unchanged.
 *
 * The new Converts are not marked here: ov::pass::MarkCompressedFloatConstants, run after this pass,
 * marks them as decompression and disables constant folding on them and their Constants.
 *
 * Run by normalize() when the decoder reports a default float type other than f32
 * (TorchDecoder::get_default_float_type; the torch.compile vLLM path passes vLLM's model dtype).
 */
class CanonicalizeFloatPrecision : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::CanonicalizeFloatPrecision");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::frontend::pytorch::pass
