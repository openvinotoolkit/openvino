// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// \ingroup ov_transformation_common_api
/// \brief Restores a dynamic batch dimension that framework tracing froze into a window-reverse view
/// shape, so the model can be reshaped along batch.
///
/// When a model is traced with a fixed batch, a window-reverse style view such as `x.view(B, H, W, -1)`
/// bakes `B` as a literal `Constant` in the leading position of the Reshape's shape `Concat`, while the
/// channel dimension is the single trailing `-1` (infer) slot. After the model batch is changed (e.g.
/// `model.reshape({input:[2,...]})`) the leading constant cannot propagate the new batch, so the `-1`
/// channel silently absorbs it and the data is mis-partitioned.
///
/// The pass rewrites the shape `Concat` that feeds the Reshape: the leading baked-batch `Constant`
/// becomes `Constant(-1)` (the batch is inferred from the real element count) and the trailing `-1`
/// (infer) slot becomes `Constant(channel)` (the batch-independent channel recovered statically). The
/// interior dimensions are kept.
///
/// Before:
///   ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  ┌──────┐
///   │  Constant(B)  │ │   <interior> │ │   <interior> │ │ Constant(-1) │  │ data │
///   │ leading batch │ │   (dynamic)  │ │   (dynamic)  │ │   channel    │  └──┬───┘
///   └───────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘     │
///           └────────────────┴───────┬────────┴────────────────┘             │
///                              ┌──────▼──────┐                                │
///                              │   Concat    │ axis = 0                       │
///                              │ (view shape)│                                │
///                              └──────┬──────┘                                │
///                                     └─────────────────┬─────────────────────┘
///                                                ┌──────▼──────┐
///                                                │   Reshape   │ special_zero = false
///                                                └──────┬──────┘
///                                                       ▼
///
/// After:
///   ┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  ┌──────┐
///   │ Constant(-1)  │ │   <interior> │ │   <interior> │ │Constant(chan)│  │ data │
///   │ leading batch │ │   (dynamic,  │ │   (dynamic,  │ │   channel    │  └──┬───┘
///   │   (inferred)  │ │  unchanged)  │ │  unchanged)  │ │  (static)    │     │
///   └───────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘     │
///           └────────────────┴───────┬────────┴────────────────┘             │
///                              ┌──────▼──────┐                                │
///                              │   Concat    │ axis = 0                       │
///                              │ (view shape)│                                │
///                              └──────┬──────┘                                │
///                                     └─────────────────┬─────────────────────┘
///                                                ┌──────▼──────┐
///                                                │   Reshape   │ special_zero = false
///                                                └──────┬──────┘
///                                                       ▼
///
/// The rewrite is value-preserving ONLY for window-reverse views, where the restored channel provably
/// equals the data tensor's channel dimension. The pass fires only when the shape `Concat` has a leading
/// positive-int constant, exactly one trailing `-1`, at least one dynamic interior dimension, AND the
/// channel can be recovered statically. Two value-preservation guards keep it from corrupting ordinary
/// reshapes that share the structural signature (spatial flatten `view(1, C, -1)`, attention head-merge
/// `view(1, N, -1)`, merge/split reshapes whose `-1` spans more than the data's last dim):
///   1. If the reshape's output last dim is statically known, it must equal the recovered channel.
///   2. (Direct path only — the channel was recovered from the data's OWN static last dim.) When the
///      output last dim is dynamic, guard 1 is vacuous; guard 2 requires the rewrite to merely
///      re-partition data's leading dimension and keep data's entire trailing block, which is exactly
///      the window-reverse semantics and rejects merge/split reshapes.
///
/// This lives in SmartReshape because it is a reshapeability concern (it runs inside `Model::reshape`),
/// not a framework-import concern: it operates on the already-built `ov::Model` and is framework-agnostic.
///
/// It is a ModelPass (not a MatcherPass) because window-reverse uses two chained views: the second view's
/// data is the permuted output of the first, whose channel OV shape inference does NOT propagate through
/// the permute (the spatial value-bounds collapse the permuted output to fully dynamic). The pass recovers
/// the channel itself with a deterministic two-phase walk-back: COLLECT iterates the ops in topological
/// order and, for each structurally matching reshape, recovers the channel by walking back from its data;
/// REWRITE then replays the recorded rewrites.
///
/// Channel recovery walk-back (the two chained window-reverse views):
///
///        data [?,8,8,180]                          (static last dim = 180)
///              │
///        ┌─────▼──────┐  Reshape_1 (1st view)   ── channel resolved DIRECTLY from data's
///        │  Reshape   │  out [?,?,?,8,8,180]        static last dim ──────────────► 180
///        └─────┬──────┘                                                              │
///              │  out last dim 180 still static                                      │
///        ┌─────▼──────┐  Transpose(order=[0,1,3,2,4,5])   last axis kept last        │
///        │ Transpose  │  out [?,?,?,?,?,?] (fully dynamic — bounds collapse)         │
///        └─────┬──────┘                                                              │
///              │  data last dim now DYNAMIC                                          │
///        ┌─────▼──────┐  Reshape_2 (2nd view)   ── channel resolved by WALK-BACK:    │
///        │  Reshape   │  out [?,H,W,-1]            Transpose (last-axis-preserving)   │
///        └─────┬──────┘                            then Reshape_1's recorded channel ┘
///              ▼
///
/// Reshape_1 takes the DIRECT path (its own data last dim is static) and the trailing-block guard
/// applies; Reshape_2 takes the WALK-BACK path (its data last dim is dynamic, recovered structurally
/// through the permute) and is exempt from the trailing-block guard.
class TRANSFORMATIONS_API RestoreReshapeBakedBatch : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreReshapeBakedBatch");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
