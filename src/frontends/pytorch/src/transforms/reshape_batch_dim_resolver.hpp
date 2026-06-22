// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/// \brief Restores a dynamic batch dimension that tracing froze into a Reshape's shape vector.
///
/// When a model is traced with a fixed batch, a window-reverse style view such as
/// `x.view(B, H, W, -1)` bakes `B` as a literal `Constant` in the leading position of the
/// Reshape's shape `Concat`, while the channel dimension is the single `-1` (infer) slot.
/// After the model batch is changed (e.g. `model.reshape({input:[2,...]})`) the leading
/// constant cannot propagate the new batch, so the `-1` channel silently absorbs it and the
/// data is mis-partitioned.
///
/// This pass detects that shape `Concat` (leading positive-int constant, a single trailing
/// `-1`, at least one dynamic interior dimension, dynamic data batch) and rebuilds it so the
/// leading dimension becomes the `-1` (inferred from the real element count) and the former
/// `-1` channel slot is pinned to the data tensor's last dimension via `Gather(ShapeOf(data),
/// -1)`. The rewrite is a no-op at the traced batch and tracks the real batch afterwards.
class ReshapeBatchDimResolver : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::ReshapeBatchDimResolver");
    ReshapeBatchDimResolver();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
