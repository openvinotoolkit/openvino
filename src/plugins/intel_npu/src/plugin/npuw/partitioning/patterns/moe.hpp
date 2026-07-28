// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

namespace patterns {
namespace moe {

// MoE pattern matching keywords
constexpr const char* ROUTER_TAG = "router";
constexpr const char* EXPERT_TAG = "expert";

// MoE layer name patterns for model detection
constexpr const char* MLP_ROUTER_NAME = ".mlp.router";
constexpr const char* MLP_EXPERT_NAME = ".mlp.expert";

// RT info key used to propagate the Router's K value from pattern-matching
// callbacks to the partition stage (written by Router matchers, read by
// PartitioningCallbacks::find_node_with_rt_info).
constexpr const char* RT_INFO_MOE_K = "npuw_moe_k";

class GPTOSSExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::GPTOSSExpert");
    static constexpr const char* pattern_name() {
        return "GPTOSSExpert";
    }
    static constexpr const char* isolation_tag() {
        return EXPERT_TAG;
    }
    static constexpr const char* group_name() {
        return "moe";
    }
    GPTOSSExpert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class GPTOSSRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::GPTOSSRouter");
    static constexpr const char* pattern_name() {
        return "GPTOSSRouter";
    }
    // NOTE: GPTOSSRouter does NOT isolate any nodes.  It only tags the matched
    // TopK with RT_INFO_MOE_K.  This tag is the ISOL_PRESETS lookup key used by
    // the pattern registry ("P:GPTOSSRouter/router") — not an isolation target.
    static constexpr const char* isolation_tag() {
        return ROUTER_TAG;
    }
    static constexpr const char* group_name() {
        return "moe";
    }
    // NOTE: isol_tag is accepted for macro-call uniformity but is intentionally unused
    //       (Router nodes are not isolated; only K is extracted via RT_INFO_MOE_K).
    GPTOSSRouter(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Qwen3Expert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::Qwen3Expert");
    static constexpr const char* pattern_name() {
        return "Qwen3Expert";
    }
    static constexpr const char* isolation_tag() {
        return EXPERT_TAG;
    }
    static constexpr const char* group_name() {
        return "moe";
    }
    Qwen3Expert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Qwen3Router : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::Qwen3Router");
    static constexpr const char* pattern_name() {
        return "Qwen3Router";
    }
    // NOTE: Qwen3Router does NOT isolate any nodes.  It only tags the matched
    // TopK with RT_INFO_MOE_K.  This tag is the ISOL_PRESETS lookup key used by
    // the pattern registry ("P:Qwen3Router/router") — not an isolation target.
    static constexpr const char* isolation_tag() {
        return ROUTER_TAG;
    }
    static constexpr const char* group_name() {
        return "moe";
    }
    // NOTE: isol_tag is accepted for macro-call uniformity but is intentionally unused
    //       (Router nodes are not isolated; only K is extracted via RT_INFO_MOE_K).
    Qwen3Router(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace moe
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
