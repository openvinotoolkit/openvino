// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RestoreReshapeBakedBatch;

}  // namespace pass
}  // namespace ov

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
/// equals the data tensor's channel dimension. The pass matches the exact window-reverse structure — a
/// pair of chained views separated by a last-axis-preserving `Transpose` — and both views must have a
/// shape `Concat` with a leading positive-int constant, exactly one trailing `-1`, and at least one
/// dynamic interior dimension. Two value-preservation guards keep it from corrupting ordinary reshapes
/// that share the structural signature (spatial flatten `view(1, C, -1)`, attention head-merge
/// `view(1, N, -1)`, merge/split reshapes whose `-1` spans more than the data's last dim):
///   1. If a reshape's output last dim is statically known, it must equal the recovered channel.
///   2. Applied whenever a reshape's OWN data last dim is static (independently of guard 1): it requires
///      the rewrite to merely re-partition the data's leading dimension and keep the data's entire
///      trailing block, which is exactly the window-reverse semantics and rejects merge/split reshapes.
///      For the window-reverse chain the output last dim is dynamic, so guard 1 is vacuous there and
///      guard 2 does the real work.
///
/// This lives in SmartReshape because it is a reshapeability concern (it runs inside `Model::reshape`),
/// not a framework-import concern: it operates on the already-built `ov::Model` and is framework-agnostic.
///
/// Window-reverse uses two chained views: the second view's data is the permuted output of the first, and
/// OV shape inference does NOT propagate the channel through the permute (the spatial value-bounds collapse
/// the permuted output to fully dynamic). The matcher therefore keys on the whole two-view chain and
/// recovers the channel from the INNER view's static data last dim; both views are rewritten together in a
/// single match, so no recorded state or traversal order is needed.
///
/// Matched chain (the two window-reverse views):
///
///        data [?,8,8,180]   (static last dim = 180)
///              │
///        ┌─────▼──────┐  Reshape_1 (INNER / 1st view): its channel is the static last dim (180) of
///        │  Reshape   │  ITS OWN data. Its output last dim is DYNAMIC (out [?,?,?,8,8,?]).
///        └─────┬──────┘
///              │
///        ┌─────▼──────┐  Transpose(order=[0,1,3,2,4,5]): last axis kept last, so the inner view's
///        │ Transpose  │  channel is also the outer view's channel. out [?,?,?,?,?,?] (bounds collapse).
///        └─────┬──────┘
///              │  data last dim is now DYNAMIC
///        ┌─────▼──────┐  Reshape_2 (OUTER / 2nd view): matched as the chain root; its channel is taken
///        │  Reshape   │  from the inner view's static data last dim (180). out [?,H,W,-1].
///        └─────┬──────┘
///              ▼
///
/// Reshape_1 has a static data last dim, so the trailing-block guard applies to it; Reshape_2's data last
/// dim is dynamic (it comes through the permute), so it is exempt from the trailing-block guard.
class ov::pass::RestoreReshapeBakedBatch : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RestoreReshapeBakedBatch");
    RestoreReshapeBakedBatch();
};
