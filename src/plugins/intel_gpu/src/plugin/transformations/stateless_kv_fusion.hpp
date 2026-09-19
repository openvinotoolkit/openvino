// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace ov::intel_gpu {

/// The diagrams below show one key/value cache. The same pattern is matched
/// independently for key and value, and the length/index helper subgraphs are
/// abbreviated as `len`, `past_len`, and `position_ids`.
///
/// 1. Dynamic shape cache update with Slice and Concat
///     Before fusion:                                      After fusion:
///     ┌───────────┐      ┌───────────┐                    ┌───────────┐      ┌───────────┐
///     │ Parameter │      │ new_token │                    │ Parameter │      │ new_token │
///     │ (past_kv) │      └─────┬─────┘                    │ (past_kv) │      └─────┬─────┘
///     └─────┬─────┘            │                          └─────┬─────┘            │
///           │                  │                                │                  │
///      ┌────┴─────┐            │                                │    ┌─────────────┴─────────────┐
///      │   Slice  │            │                                └────┤        StatelessKV        │<── len
///      │(past_len)│            │                                     │          (axis )          │
///      └────┬─────┘            │                                     └───────┬────────────┬──────┘
///           │   ┌───────────┐  │                                             │0           │1
///           └───┤   Concat  ├──┘                                        ┌────┴────┐   ┌───┴─────┐
///               └────┬──────┘                                           │ Result  │   │ SomeOp  │
///                    │                                                  │(present)│   │ (SDPA)  │
///         ┌──────────┴─────────┐                                        └─────────┘   └─────────┘
///    ┌────┴─────┐          ┌───┴─────┐          
///    │  Result  │          │ SomeOp  │
///    │(present) │          │ (SDPA)  │
///    └──────────┘          └─────────┘
///
/// 2. Static shape cache update with ScatterUpdate (mask will also be trimmed)
///     Before fusion:                                      After fusion:
///     ┌───────────┐      ┌───────────┐                    ┌───────────┐      ┌───────────┐
///     │ Parameter │      │ new_token │                    │ Parameter │      │ new_token │
///     │ (past_kv) │      └─────┬─────┘                    │ (past_kv) │      └─────┬─────┘
///     └─────┬─────┘            │                          └─────┬─────┘            │
///           │                  │                                │                  │
///           │             ┌────┴─────┐                          │    ┌─────────────┴─────────────┐
///           └─────────────┤ Scatter  │<── position_ids          └────┤         StatelessKV       │<── len
///                         │ Update   │<── axis                       │            (axis)         │<── position_ids 
///                         └────┬─────┘                               └───────┬────────────┬──────┘
///                              │                                             │0           │1
///                 ┌────────────┴─────────┐                              ┌────┴────┐   ┌───┴─────┐
///            ┌────┴─────┐            ┌───┴─────┐                        │ Result  │   │ SomeOp  │
///            │  Result  │            │ SomeOp  │                        │(present)│   │ (SDPA)  │
///            │(present) │            │ (SDPA)  │                        └─────────┘   └─────────┘
///            └──────────┘            └─────────┘
///
/// 3. Static shape cache update with ScatterUpdate and VariadicSplit before SDPA
///     Before fusion:                                      After fusion:
///     ┌───────────┐      ┌───────────┐                    ┌───────────┐      ┌───────────┐
///     │ Parameter │      │ new_token │                    │ Parameter │      │ new_token │
///     │ (past_kv) │      └─────┬─────┘                    │ (past_kv) │      └─────┬─────┘
///     └─────┬─────┘            │                          └─────┬─────┘            │
///           │                  │                                │                  │
///           │             ┌────┴─────┐                          │    ┌─────────────┴─────────────┐
///           └─────────────┤Scatter   │<── position_ids          └────┤         StatelessKV       │<── len
///                         │Update    │<── axis                       │           (axis)          │<── position_ids 
///                         └────┬─────┘                               └───────┬────────────┬──────┘
///                              │                                             │0           │1
///                 ┌────────────┴───────────┐                            ┌────┴────┐   ┌───┴─────┐
///            ┌────┴─────┐          ┌───────┴───────┐                    │ Result  │   │ SomeOp  │
///            │  Result  │          │ VariadicSplit │                    │(present)│   │ (SDPA)  │
///            │(present) │          │   [len, -1]   │                    └─────────┘   └─────────┘
///            └──────────┘          └───────┬───────┘
///                                          │0
///                                     ┌────┴────┐
///                                     │ SomeOp  │
///                                     │ (SDPA)  │
///                                     └─────────┘
///
/// StatelessKV's output0 is the complete present_kv for Result, output1 is the valid view consumed by SDPA.
/// In case2, SDPA will recieve a trimmed view and mask will also be trimmed with present_len.
class StatelessKVFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("StatelessKVFusionMatcher");
    StatelessKVFusionMatcher();

private:
    struct CachedNodes {
        std::shared_ptr<ov::Node> total_seqlen;
        std::shared_ptr<ov::Node> seqlens_k;
        ov::Output<ov::Node> present_kv_len;

        struct TrimmedMask {
            int64_t new_token_len = 0;
            ov::Output<ov::Node> original_mask;
            std::shared_ptr<ov::Node> trimmed_mask;
        };
        std::vector<TrimmedMask> trimmed_masks;
    };
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<CachedNodes>> m_cache;
    std::set<ov::Output<ov::Node>> m_trimmed_masks;
};

class StatelessKVFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("StatelessKVFusion");
    StatelessKVFusion();

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}   // namespace ov::intel_gpu
