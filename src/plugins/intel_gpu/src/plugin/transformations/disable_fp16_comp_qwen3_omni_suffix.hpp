// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps the numerically sensitive suffix of Qwen3-Omni's stateful
 *        code predictor in FP32 when the model is compiled with an FP16 hint.
 *
 * The protected region starts after layer 2 attention and contains layer 2
 * MLP, layers 3-4, final normalization, head, and sampling. Stateful KV cache
 * variables remain FP16. Explicit Convert nodes at Assign inputs form the
 * FP32-compute to FP16-cache boundary.
 *
 * Activation is based only on a strict graph contract: five serial Gated-MLP
 * transformer blocks, five SDPA nodes, ten state pairs, and the predictor's
 * code/embedding output shapes. Model, node, and tensor names are ignored.
 */
class DisableFP16CompForQwen3OmniCodePredictorSuffix : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DisableFP16CompForQwen3OmniCodePredictorSuffix");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu