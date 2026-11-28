// Copyright (C) 2018-2026 Intel Corporation
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
class KVCacheFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KVCacheFusionMatcher");
    KVCacheFusionMatcher();
};

/// The block index key cache of a stateful QSA indexer, as FuseQSAIndexer(padding = true) leaves it: the cache gets a head
/// axis and is grown by zero blocks to floor(max(key_length) / key_block_size) blocks before QSAIndexer, whose updated cache
/// is squeezed and sliced back for the Assign.
///
///   ReadValue -> [Gather(beam_idx)] -> Unsqueeze -> Concat(zero blocks) -> QSAIndexer -> Squeeze -> Slice -> [Convert] -> Assign
///
/// The growth becomes a KVCache appending the zero blocks to the variable, which its preallocation makes in place, and
/// QSAIndexerInPlace writes the completed blocks into that buffer instead of returning a copy:
///
///   ReadValue -> KVCache(zero blocks[, beam_idx]) -> QSAIndexerInPlace
///
/// The KVCache holds max(past, needed) blocks, where needed = max(key_length) / key_block_size is the block count the Slice
/// keeps: QSAIndexer only reads the blocks completed by key_length, so it equals the sliced cache wherever it is read.
/// The two differ only when max(key_length) shrinks without a reset (a caller rolling the sequence back): the variable
/// keeps its capacity, so get_state() returns the `needed` blocks of the unfused model followed by the stale blocks of
/// the longer sequence, which are overwritten as the sequence grows again before being read.
/// A beam_idx Gather becomes an indirect KVCache whose beam_idx must be the identity (see
/// op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX), as the indexer reads the cache rows as they are.
class QSAIndexerBlockCacheFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSAIndexerBlockCacheFusionMatcher");
    QSAIndexerBlockCacheFusionMatcher();
};

class KVCacheFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("KVCacheFusion");
    KVCacheFusion();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};


}   // namespace ov::intel_gpu
