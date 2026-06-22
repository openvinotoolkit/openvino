// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps all RMS normalization nodes in FP32.
 *
 * RMS (Root Mean Square normalization) computes rsqrt(mean(x²)) which is
 * numerically unstable in FP16: when preceding MatMul outputs exceed the FP16
 * range (>65504), they overflow to INF, and rsqrt(INF)=0 leads to INF*0=NaN
 * per IEEE 754.  Keeping RMS in FP32 prevents this chain of corruption.
 *
 * This pass is a catch-all for RMS nodes not already protected by more specific
 * pattern-based passes (e.g. DisableFP16CompForGemma3RMSPattern).  It skips
 * nodes that already have the disable_conversion mark set.
 *
 * Performance impact is negligible: RMS is element-wise and memory-bound,
 * not compute-bound like MatMul.
 */
class DisableFP16CompForRMSNormBlock : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("DisableFP16CompForRMSNormBlock");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
