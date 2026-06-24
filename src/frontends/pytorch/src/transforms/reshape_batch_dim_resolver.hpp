// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/// \brief Restores a dynamic batch dimension that tracing froze into a window-reverse view shape.
///
/// When a model is traced with a fixed batch, a window-reverse style view such as
/// `x.view(B, H, W, -1)` bakes `B` as a literal `Constant` in the leading position of the
/// Reshape's shape `Concat`, while the channel dimension is the single trailing `-1` (infer)
/// slot. After the model batch is changed (e.g. `model.reshape({input:[2,...]})`) the leading
/// constant cannot propagate the new batch, so the `-1` channel silently absorbs it and the
/// data is mis-partitioned.
///
/// The pass rewrites the shape `Concat` that feeds the Reshape: the leading baked-batch
/// `Constant` becomes `Constant(-1)` (so the batch is inferred from the real element count) and
/// the trailing `-1` (infer) slot becomes `Constant(channel)` (the batch-independent channel that
/// was baked into the data, recovered statically — see below). The interior dimensions are kept.
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
/// The rewrite — leading dimension becomes `-1` (inferred from the real
/// element count) and the former `-1` channel is pinned to its (batch-independent) value as a
/// `Constant` — is value-preserving ONLY for window-reverse views, where the restored channel
/// provably equals the data tensor's channel dimension. The pass fires only when the matched
/// shape `Concat` has a leading positive-int constant, exactly one trailing `-1`, at least one
/// dynamic interior dimension, AND the channel can be recovered statically (see below). That last
/// requirement excludes ordinary reshapes such as spatial flatten `view(1, C, -1)` or attention
/// head-merge `view(1, N, -1)`, whose `-1` is a product of several dimensions and whose data
/// (fed from a Parameter) has a dynamic last dim that cannot be recovered. At the traced batch the
/// rewritten shape reproduces the original element count; after `model.reshape` it tracks the real
/// batch.
///
/// Two value-preservation guards keep the rewrite from corrupting ordinary reshapes that happen to
/// share the structural signature:
///   1. If the reshape's output last dim is statically known, it must equal the recovered channel.
///   2. (Direct path only — the channel was recovered from the data's OWN static last dim.) When the
///      output last dim is dynamic, guard 1 is vacuous, so a reshape whose `-1` spans more than data's
///      last dim — e.g. `Linear(D, D)` then `view(1, T//2, -1)` (here `-1 == 2*D`, a head-merge over a
///      static-`D` tensor with a dynamic interior) — would otherwise be rewritten and corrupt the result
///      even at the traced batch. Guard 2 requires the rewrite to merely re-partition data's leading
///      dimension and keep data's entire trailing block (output trailing `rank_data-1` dims equal data's
///      trailing dims), which is exactly the window-reverse semantics and rejects merge/split reshapes.
///
/// This is a ModelPass rather than a MatcherPass because window-reverse uses two chained views:
/// the second view (`view(B, H, W, -1)`) takes its data from the first view (`view(B, ..., -1)`)
/// through a permute, so its data last dimension is dynamic at the time the pass runs and OV shape
/// inference does NOT propagate the first view's static channel through that permute (the spatial
/// value-bounds collapse the permuted output to fully dynamic). The pass therefore recovers the
/// channel itself with a deterministic two-phase walk-back: COLLECT iterates the ops in topological
/// order and, for each structurally matching reshape, recovers the channel by walking back from its
/// data — directly if the data's last dim is static, through a last-axis-preserving `Transpose`
/// (order ends in `rank-1`), or through a reshape already selected for rewrite — then REWRITE
/// replays the recorded rewrites. A static-output-channel value-preservation guard rejects any case
/// where the recovered channel disagrees with the reshape's statically inferred output channel.
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
class ReshapeBatchDimResolver : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::ReshapeBatchDimResolver");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
