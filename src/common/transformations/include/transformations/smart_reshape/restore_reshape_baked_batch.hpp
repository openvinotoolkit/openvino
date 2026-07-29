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
/// When a model is traced with a fixed batch, a window-reverse view such as `x.view(B, H, W, -1)` bakes
/// `B` as a literal leading `Constant` in the Reshape's shape `Concat`, while the channel is the single
/// trailing `-1` slot. After the batch is changed (`model.reshape(...)`) the baked constant cannot
/// propagate it, so the `-1` channel silently absorbs it and the data is mis-partitioned. The pass
/// rewrites the shape `Concat` to relax the leading `Constant(B)` to `Constant(-1)` (batch inferred) and
/// pin the trailing `-1` to `Constant(channel)` (recovered statically); the interior is kept.
///
///   Before:  Concat[ Constant(B), <interior...>, Constant(-1) ] ──► Reshape (special_zero=false)
///   After:   Concat[ Constant(-1), <interior...>, Constant(chan) ] ──► Reshape
///
/// The rewrite is value-preserving ONLY for window-reverse views, where the restored channel provably
/// equals the data's channel dim. Window-reverse emits two chained views separated by a last-axis-
/// preserving `Transpose`; OV shape inference does NOT propagate the channel through that permute (the
/// spatial value-bounds collapse the permuted output to fully dynamic), so the matcher keys on the whole
/// two-view chain and recovers the channel from the INNER view's static data last dim. Both views are
/// rewritten together in a single match, so no recorded state or traversal order is needed.
///
/// Each view must have a shape `Concat` with a leading positive-int constant, exactly one trailing `-1`,
/// and at least one dynamic interior dim. Two value-preservation guards keep the pass off ordinary
/// reshapes that share this signature:
///   1. If a reshape's output last dim is static, it must equal the recovered channel.
///   2. If a reshape's own data last dim is static, the rewrite must merely re-partition the data's
///      leading dim and keep its entire trailing block (the window-reverse semantics). For the chain the
///      output last dim is dynamic, so guard 1 is vacuous and guard 2 does the work.
///
/// It lives in SmartReshape because it is a reshapeability concern (it runs inside `Model::reshape`) on
/// the already-built framework-agnostic `ov::Model`, not a framework-import concern.
class ov::pass::RestoreReshapeBakedBatch : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RestoreReshapeBakedBatch");
    RestoreReshapeBakedBatch();
};
