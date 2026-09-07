// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Duplicate shared KV broadcast chains to enable per-SDPA ATTN isolation.
 *
 * Some models (e.g. Gemma-4) share a single accumulated KV tensor across multiple
 * SDPA nodes by fanning out through a Reshape node:
 *
 *   Concat(past_K_blocks…, current_K)
 *       → [Unsqueeze] → [Broadcast] → Reshape ─┬─→ SDPA_L13
 *                                              ├─→ SDPA_L15
 *                                              ├─→ SDPA_L16
 *                                              └─→ …
 *
 * The high fan-out prevents NPUW's ATTN isolation patterns (TagSDPA /
 * SDPADecomposed) from tagging each SDPA independently, because every tagging
 * attempt would claim the shared Reshape (and its ancestors) as part of its
 * own partition, creating conflicting assignments.
 *
 * This pass duplicates the [Convert →] Concat → [Unsqueeze] → [Broadcast] → Reshape
 * chain for every extra consumer beyond the first, giving each SDPA its own
 * local chain:
 *
 *   same past_K_blocks… → [new_Convert] → new_Concat_L15 → … → new_Reshape_L15 → SDPA_L15
 *   same past_K_blocks… → [new_Convert] → new_Concat_L16 → … → new_Reshape_L16 → SDPA_L16
 *   …
 *
 * All duplicate chains share the same bare Parameter (block) nodes and the same
 * current_K subgraph.  However, any Convert nodes that directly wrap past-KV
 * Parameters are cloned independently for each new chain.  Without this,
 * subgraph-isolation passes (e.g. SDPADecomposed) would pull the shared
 * Convert nodes into the first SDPA's subgraph group, creating spurious
 * cross-subgraph pass-through outputs on the producer subgraph and mismatching
 * the output count expected by HFA tile compilation.
 */
class DuplicateSharedKVConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::pass::DuplicateSharedKVConcat");
    DuplicateSharedKVConcat();
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
