// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Fuses the frontend's atan2 decomposition (~10 ops) into a single
 *        ov::intel_gpu::op::Atan2, which lowers to one cldnn::eltwise call.
 *
 * The frontend (translate_atan2_util) lowers torch.atan2 / aten::angle into
 * Atan(div) plus a 3-level Select chain that handles ±π / ±π/2 quadrant
 * adjustments. This pass collapses the chain back into a single Atan2.
 *
 *   Before                                  After
 *   ------                                  -----
 *   lhs (y)   rhs (x)                       lhs (y)   rhs (x)
 *     │         │                             │         │
 *     └─►Divide / Multiply(_,Pow(_,-1))       └────┬────┘
 *            │                                     ▼
 *          Atan                              Atan2 (eltwise)
 *            │                                     │
 *       Select_1 ── (rhs<0, lhs>=0): atan±π        ▼
 *            │                                downstream
 *       Select_2 ── (rhs>0): atan
 *            │
 *       Select_3 (root) ── (rhs==0): ±π/2
 *            │
 *            ▼
 *        downstream
 *
 * Targets: BiLSTM/STFT-style atan2 in TTS decoders (Kokoro).
 */
class FuseAtan2Decomposed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseAtan2Decomposed");
    FuseAtan2Decomposed();
};

}  // namespace ov::intel_gpu
