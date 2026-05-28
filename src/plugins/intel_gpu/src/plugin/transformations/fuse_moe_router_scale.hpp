// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuse a per-expert routing scale (Const[N] → Gather(topk_indices) → Multiply)
/// that sits between MoERouterFused and MOECompressed.input(1) by folding the
/// per-expert scale into w2_scale (input 10 of MOECompressed).
///
/// Matched pattern:
///   routing  ──► Multiply ──► MOECompressed.input(1)
///   Gather ──────────────┘
///   (Const[N] → Gather(topk_indices, axis))
///
/// The per-expert scale constant is unsqueezed to [N, 1, 1, ...] and constant-
/// folded into w2_scale so that no extra eltwise remains in the graph.
class FuseMoERouterScale : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoERouterScale");
    FuseMoERouterScale();
};

}  // namespace ov::intel_gpu
