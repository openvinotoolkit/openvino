// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

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
class IndirectKVCache : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("IndirectKVCache", "0");
    IndirectKVCache();
};

}   // namespace intel_gpu
}   // namespace ov
