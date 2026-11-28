// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// @brief Splits a fused projection followed by a head split into one MatMul per used part:
///     Gather(Transpose(Reshape(MatMul(x, W), [B, L, P, H, D]), [2, 0, 3, 1, 4]), k, axis 0)
/// becomes
///     Transpose(Reshape(MatMul(x, W[:, k*H*D : (k+1)*H*D]), [B, L, H, D]), [0, 2, 1, 3])
/// so the 5D transpose is removed. The constant (e.g. compressed) weight is sliced along the last axis, so its size
/// doesn't change. Applied when all consumers of the Transpose are such Gathers with distinct indices.
class QKVSplitMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QKVSplitMatMul");
    QKVSplitMatMul();
};

}  // namespace ov::intel_gpu