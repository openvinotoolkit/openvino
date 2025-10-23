// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov::pass {

/**
 * @brief MultiMatcher applies multiple independent pattern matchers across the graph.
 *
 * Each registered pattern is independently matched across all graph nodes.
 * Matches are grouped by pattern root and passed to a single callback.
 *
 * This is especially useful for repeated blocks like attention heads, Q/K/V projections,
 * and residual connections.
 *
 * Match repeated Q/K/V branches
 *
 *   ┌──────────── Pattern 1 ─────────────┐      ┌──────────── Pattern 2 ─────────────┐
 *   │                                    │      │                                    │
 *   │ Input ──► MatMul_Q ──► Add_Q       │      │ Input ──► MatMul_K ──► Add_K       │
 *   └────────────────────────────────────┘      └────────────────────────────────────┘
 *                         │                                  │
 *                         ▼                                  ▼
 *                    concat_qkv                         concat_qkv
 *                         │                                  │
 *                         ▼                                  ▼
 *                       SDPA (shared)
 *
 * Each Q/K/V branch is matched separately using a different pattern.
 *
 * After matching, callback receives:
 *
 * matches = {
 *   Pattern1: [MatchQ1, MatchQ2, ...],
 *   Pattern2: [MatchK1, MatchK2, ...],
 *   Pattern3: [MatchV1, MatchV2, ...]
 * }
 */
class OPENVINO_API MultiMatcher : public ov::pass::ModelPass {
public:
    using Callback =
        std::function<void(const std::unordered_map<std::shared_ptr<Node>, std::vector<pattern::PatternValueMap>>&)>;

    OPENVINO_RTTI("MultiMatcher", "0", ModelPass);

    explicit MultiMatcher(const std::string& name = "MultiMatcher");

    /**
     * @brief Register multiple patterns with a unified callback
     * @param patterns  Vector of pattern root nodes
     * @param callback  Callback applied to all matches grouped by pattern
     * @param strict    Whether to use strict mode in Matcher
     */
    void register_patterns(const std::vector<std::shared_ptr<Node>>& patterns, Callback callback, bool strict = false);

    /**
     * @brief Run all matchers once over the model
     * @return true if any matches were found and callback invoked
     */
    bool run_on_model(const std::shared_ptr<Model>& model) override;

private:
    struct PatternEntry {
        ov::Output<ov::Node> pattern;
        std::shared_ptr<ov::Node> root_ptr;
        bool strict_mode = false;
    };

    std::string m_name;
    Callback m_callback;
    std::vector<PatternEntry> m_patterns;
    std::unordered_set<Node*> m_matched_nodes;
    std::unordered_set<Node*> m_all_roots;
};

}  // namespace ov::pass
