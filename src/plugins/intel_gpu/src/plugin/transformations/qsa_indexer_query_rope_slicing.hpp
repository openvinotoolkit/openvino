// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/// Folds the partial rotary embedding of a QSA indexer query into one RoPE over the whole head, so that
/// ov::pass::FuseQSAIndexerQuery can absorb it into the indexer (the GPU indexer requires query_rope).
///
/// The GPU plugin disables RoPEFusionIOSlicing because its RoPE kernels do not copy the channels past rotary_ndims,
/// so an exported partial RoPE stays split:
///
///   q [B, L, H, D] ─► Transpose(0, 2, 1, 3) ─► VariadicSplit(-1, {rotary, D - rotary})
///     ─► Concat(RoPE(rotary part), pass-through part, -1) ─► ... ─► indexer query
///
/// This pass applies the IOSlicing rewrite only to that chain, as RoPE(q, input_trans0213, rotary_ndims = rotary) (or
/// a RoPE over the head-first query as it is, when it is not a Transpose, e.g. the single-token paged query whose
/// Transpose became a Reshape):
/// the result only feeds the indexer query (through Transpose / Reshape nodes), where FuseQSAIndexerQueryRoPE absorbs
/// it, so it never reaches the RoPE kernels. Other partial rotary embeddings (e.g. of the attention query and key)
/// are left as they are.
class QSAIndexerQueryRoPESlicing : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSAIndexerQueryRoPESlicing");
    QSAIndexerQueryRoPESlicing();
};

}  // namespace ov::intel_gpu
