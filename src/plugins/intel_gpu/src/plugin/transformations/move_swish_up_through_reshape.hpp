// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// \brief Hoists a Swish (SiLU) above a single-consumer Reshape.
///
/// Rewrites `Swish(Reshape(x))` into `Reshape(Swish(x))`. Swish is elementwise and a
/// Reshape only re-indexes, so this is value-for-value identical -- it changes nothing
/// numerically, only which node the activation is adjacent to.
///
/// Motivation: in Qwen3.5 / Gated-DeltaNet the SiLU that follows the causal conv is
/// separated from its producer by a Reshape that PagedCausalConv1DFusion inserts to
/// restore the 3D shape. The same shape holds for the `in_proj_z` gate. The GPU plugin's
/// prepare_primitive_fusing only fuses an activation into its *direct* dependency, so that
/// interposed Reshape is what keeps the SiLU as a standalone kernel -- one that re-reads
/// and re-writes the whole activation tensor at the device's memory-bandwidth ceiling.
/// Hoisting the Swish makes it adjacent to the conv / FullyConnected again so the existing
/// fusing pass can absorb it.
///
/// ov::pass::MoveEltwiseUpThroughDataMov does not cover this: it matches
/// UnaryElementwiseArithmetic / BinaryElementwiseArithmetic, and ov::op::v4::Swish derives
/// from ov::op::Op directly (it takes an optional beta input), so it is never matched.
class MoveSwishUpThroughReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveSwishUpThroughReshape");
    MoveSwishUpThroughReshape();
};

}  // namespace ov::intel_gpu
