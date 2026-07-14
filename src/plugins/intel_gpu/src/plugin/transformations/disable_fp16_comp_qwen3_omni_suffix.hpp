// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformations/convert_precision.hpp"

namespace ov::intel_gpu {

/**
 * @brief Runs GPU floating-point precision conversion with the Trial-3
 *        protection required by Qwen3-Omni's stateful code predictor.
 *
 * The protected regions are the first layer's fused Q/K/V projection and a
 * suffix starting after layer 2 attention that contains layer 2 MLP, layers
 * 3-4, final normalization, head, and sampling. The five structurally matched
 * SDPA reductions, their Q/K/V compute frontiers, and stateful KV caches are
 * also protected. The first two MLPs remain FP16.
 *
 * Before the ordinary ConvertPrecision implementation runs, this pass marks
 * the sensitive nodes. A successful match selects a conversion configuration
 * that leaves this model's Variables in FP32; a failed match selects the
 * ordinary GPU configuration that converts Variables normally. This keeps all
 * CodePredictor-specific decisions out of generic ConvertPrecision.
 *
 * Activation is based only on a strict graph contract: a connected five-layer
 * attention-residual/Gated-MLP stack, structurally paired KV-cache state, and
 * the predictor's code/embedding outputs. Unrelated operation/state counts and
 * model, node, and tensor names are ignored.
 */
class ConvertPrecisionForQwen3OmniCodePredictor : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertPrecisionForQwen3OmniCodePredictor");
    explicit ConvertPrecisionForQwen3OmniCodePredictor(const precisions_map& precisions);

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    precisions_map m_precisions;
};

}  // namespace ov::intel_gpu