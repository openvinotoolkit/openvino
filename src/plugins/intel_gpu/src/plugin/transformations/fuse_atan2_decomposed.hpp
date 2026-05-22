// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Fuses the frontend's atan2 decomposition (~10 ops) into a single
 *        ov::intel_gpu::op::Atan2, which lowers to one cldnn::eltwise call.
 *
 * The frontend lowers torch.atan2 / aten::angle as:
 *
 *      div   = Divide(lhs, rhs)             // (or Multiply(lhs, Power(rhs, -1))
 *      atan  = Atan(div)                    //  after ConvertDivide)
 *      Sel1  = Select(LogicalAnd(rhs<0, lhs>=0), atan+π, atan-π)
 *      Sel2  = Select(rhs>0, atan, Sel1)
 *      Sel3  = Select(LogicalOr(LogicalAnd(rhs==0, lhs>0),
 *                               LogicalAnd(rhs==0, lhs<0)),
 *                     Select(LogicalAnd(rhs==0, lhs>0), +π/2, -π/2),
 *                     Sel2)              // ← root
 *
 *  Before                          After
 *  ------                          -----
 *  lhs (y)   rhs (x)
 *    │         │                      lhs (y)   rhs (x)
 *    └─►Divide / Multiply(_,Pow)         │         │
 *           │                            └────┬────┘
 *         Atan                                ▼
 *           │ ── Add+π / Add-π          Atan2 (eltwise)
 *           │
 *      Select_1 ── LogicalAnd(rhs<0, lhs>=0)
 *           │
 *      Select_2 ── Greater(rhs, 0)
 *           │
 *      Select_3 (root) ── LogicalOr(...) of axis cases
 *           │
 *           ▼
 *      downstream
 *
 * Targets: BiLSTM/STFT-style atan2 in TTS decoders (Kokoro). Walks the model
 * directly (ModelPass) to handle type_relaxed Select nodes that wrap_type
 * matchers don't catch. Runs before IncreaseAtan2DivPrecision so that pass has
 * nothing to do once fusion succeeds.
 */
class FuseAtan2Decomposed : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseAtan2Decomposed");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace ov::intel_gpu
