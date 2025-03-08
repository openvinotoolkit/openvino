// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// 1. Trivial case (greedy search, no state initializer)
///     ┌───────────┐      ┌───────────┐                                      ┌───────────┐
///     │ ReadValue │      │  SomeOp   │                                      │  SomeOp   │
///     | (past_kv) |      | new_token |                                      | new_token |
///     └─────┬─────┘      └─────┬─────┘                                      └─────┬─────┘
///           │                  │                                                  |
///           │                  |                                                  |
///           │   ┌────────┐     │                                             ┌────┴────┐       ┌──────────┐
///           └───┤ Concat ├─────┘                    =>                       | KVCache |.......| Variable |
///               └───┬────┘                                                   └────┬────┘       └──────────┘
///                   │                                                             |
///        ┌──────────┴────────────┐                                                |
///   ┌────┴────────┐         ┌────┴──────┐                                    ┌────┴────┐
///   │  Assign     │         │  SomeOp   │                                    | SomeOp  |
///   | (present_kv |         |  (SDPA)   |                                    | (SDPA)  |
///   └─────────────┘         └───────────┘                                    └─────────┘

/// 2. With gather for beam search (or model which supports both greedy and beam search)
///     ┌───────────┐      ┌────────────┐    ┌───────────┐                   ┌───────────┐      ┌────────────┐    ┌───────────┐
///     │ ReadValue │      │ Parameter  │    │  SomeOp   │                   │ ReadValue │      │ Parameter  │    │  SomeOp   │
///     | (past_kv) |      |  beam_idx  |    | new_token |                   | (past_kv) |      |  beam_idx  |    | new_token |
///     └─────┬─────┘      └─────┬──────┘    └─────┬─────┘                   └─────┬─────┘      └─────┬──────┘    └─────┬─────┘
///           │                  │                 │                               │                  │                 │
///     ┌─────┴──────┐           |                 │                         ┌─────┴──────┐           |                 │
///     |   Gather   |───────────┘                 │                         |   Gather   |───────────┘                 │
///     └─────┬──────┘                             │                         └─────┬──────┘                             │
///           |                                    │                               |                        ┌───────────┘
///           |                                    │                               |                        |
///           │                                    │                               |                        |
///           │   ┌────────┐                       │                               |            ┌───────────┴───────┐                       ┌──────────┐
///           └───┤ Concat ├───────────────────────┘        =>                     └────────────┤      KVCache      |.......................| Variable |
///               └───┬────┘                                                                    └────┬──────────────┘                       └──────────┘
///                   │                                                                              |
///        ┌──────────┴─────────────┐                                                                |
///   ┌────┴─────────┐         ┌────┴──────┐                                                    ┌────┴────┐
///   │  Assign      │         │  SomeOp   │                                                    | SomeOp  |
///   | (present_kv) |         |  (SDPA)   |                                                    | (SDPA)  |
///   └──────────────┘         └───────────┘                                                    └─────────┘

/// 3. Similar to case 2, but with variable initializer
///     ┌─────────────────────┐                                               ┌─────────────────────┐
///     │       SomeOp        │                                               │       SomeOp        │
///     | (state initializer) |                                               | (state initializer) |
///     └─────┬───────────────┘                                               └─────┬───────────────┘
///           |                                                                     |
///     ┌─────┴─────┐      ┌────────────┐    ┌───────────┐                    ┌─────┴─────┐      ┌────────────┐    ┌───────────┐
///     │ ReadValue │      │ Parameter  │    │  SomeOp   │                    │ ReadValue │      │ Parameter  │    │  SomeOp   │
///     | (past_kv) |      |  beam_idx  |    | new_token |                    | (past_kv) |      |  beam_idx  |    | new_token |
///     └─────┬─────┘      └─────┬──────┘    └─────┬─────┘                    └─────┬─────┘      └─────┬──────┘    └─────┬─────┘
///           │                  │                 │                                │                  │                 │
///     ┌─────┴──────┐           |                 │                          ┌─────┴──────┐           |                 │
///     |   Gather   |───────────┘                 │                          |   Gather   |───────────┘                 │
///     └─────┬──────┘                             │                          └─────┬──────┘                             │
///           |                                    │                                |                        ┌───────────┘
///           |                                    │                                |                        |
///           │                                    │                                |                        |
///           │   ┌────────┐                       │                                |            ┌───────────┴───────┐                       ┌──────────┐
///           └───┤ Concat ├───────────────────────┘        =>                      └────────────┤      KVCache      |.......................| Variable |
///               └───┬────┘                                                                     └────┬──────────────┘                       └──────────┘
///                   │                                                                               |
///        ┌──────────┴────────────┐                                                                  |
///   ┌────┴────────┐         ┌────┴──────┐                                                      ┌────┴────┐
///   │  Assign     │         │  SomeOp   │                                                      | SomeOp  |
///   | (present_kv |         |  (SDPA)   |                                                      | (SDPA)  |
///   └─────────────┘         └───────────┘                                                      └─────────┘
class KVCacheFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("KVCacheFusion");
    KVCacheFusion();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
