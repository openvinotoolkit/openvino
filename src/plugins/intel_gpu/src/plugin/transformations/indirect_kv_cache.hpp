// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Merges Gather into KVCache op
///     ┌─────────────────────┐                                               ┌─────────────────────┐
///     │       SomeOp        │                                               │       SomeOp        │
///     | (state initializer) |                                               | (state initializer) |
///     └─────┬───────────────┘                                               └─────┬───────────────┘
///           |                                                                     |
///     ┌─────┴─────┐      ┌────────────┐    ┌───────────┐                    ┌─────┴─────┐     ┌────────────┐    ┌───────────┐
///     │ ReadValue │      │ Parameter  │    │  SomeOp   │                    │ ReadValue │     │ Parameter  │    │  SomeOp   │
///     | (past_kv) |      |  beam_idx  |    | new_token |                    | (past_kv) |     |  beam_idx  |    | new_token |
///     └─────┬─────┘      └─────┬──────┘    └─────┬─────┘                    └─────┬─────┘     └─────┬──────┘    └─────┬─────┘
///           │                  │                 │                                │                 │                 │
///           │                  │                 │                                |                 |      ┌──────────┘
///     ┌─────┴──────┐           |                 │                                |                 |      |
///     |   Gather   |───────────┘                 │                                |                 |      |
///     └─────┬──────┘                             │                                |                 |      |
///           |         ┌──────────────────────────┘                                |                 |      |
///           |         |                                                           |                 |      |
///           │         |                                                           |                 |      |
///           │   ┌─────┴───┐               ┌──────────┐                            |            ┌────┴──────┴───────┐                       ┌──────────┐
///           └───┤ KVCache |...............| Variable |        =>                  └────────────┤      KVCache      |.......................| Variable |
///               └───┬─────┘               └──────────┘                                         └────┬──────┬───────┘                       └──────────┘
///                   │                                                                               |      |
///                   |                                                                     kv_cache  |      | beam_table
///                   |                                                                               |      |
///              ┌────┴──────┐                                                                   ┌────┴──────┴───┐
///              │   Gemm    │                                                                   | IndirectGemm  |
///              └───────────┘                                                                   └───────────────┘
class IndirectKVCache : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("IndirectKVCache");
    IndirectKVCache();
};

class IndirectGemmOpt : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IndirectGemmOpt");
    IndirectGemmOpt();
};

class IndirectSDPAOpt : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IndirectSDPAOpt");
    IndirectSDPAOpt();
};

/// Makes the key / value KVCache of the GPU SparseSDPA op (see ConvertSparseAttentionToInternal) indirect, as IndirectSDPAOpt does for SDPA, so that
/// the caches are not gathered by beam_idx on every step and can be compressed by KVCacheCompression.
/// The sparse attention kernels do not read a beam table yet: the caches are marked with
/// op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX and the infer request rejects a beam_idx that is not the identity, i.e. beam
/// search is not supported for these layers (greedy decoding and sampling are).
class IndirectSparseAttentionOpt : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IndirectSparseAttentionOpt");
    IndirectSparseAttentionOpt();
};

}   // namespace ov::intel_gpu
