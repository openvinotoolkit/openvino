// Copyright (C) 2024 Intel Corporation
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

}   // namespace ov::intel_gpu
