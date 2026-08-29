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
constexpr const char* MLP_ROUTER_NAME = ".router";
constexpr const char* MLP_EXPERT_NAME = ".expert";

// RT info key used to propagate the Router's K value from pattern-matching
// callbacks to the partition stage (written by Router matchers, read by
// PartitioningCallbacks::find_node_with_rt_info).
constexpr const char* RT_INFO_MOE_K = "npuw_moe_k";

// Boilerplate macros for MoE MatcherPass classes.
//
// Each MoE pass class requires four identical declarations: RTTI, pattern_name(),
// isolation_tag(), and group_name(). These macros centralise that boilerplate so
// that adding a new model variant requires only a constructor declaration in this
// header and a corresponding definition in moe.cpp.
//
// MOE_EXPERT_STATIC_INFO  — for Expert passes that isolate matched subgraph nodes.
// MOE_ROUTER_STATIC_INFO  — for Router passes that only extract K via RT_INFO_MOE_K
//                           and do NOT isolate any nodes. The isol_tag constructor
//                           argument is accepted for call-site uniformity but is
//                           intentionally unused in all Router implementations.
// clang-format off
#define MOE_EXPERT_STATIC_INFO(ClassName)                                                    \
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::" #ClassName);                          \
    static constexpr const char* pattern_name()  { return #ClassName; }                      \
    static constexpr const char* isolation_tag() { return EXPERT_TAG; }                      \
    static constexpr const char* group_name()    { return "moe"; }

#define MOE_ROUTER_STATIC_INFO(ClassName)                                                    \
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::" #ClassName);                          \
    static constexpr const char* pattern_name()  { return #ClassName; }                      \
    static constexpr const char* isolation_tag() { return ROUTER_TAG; }                      \
    static constexpr const char* group_name()    { return "moe"; }
// clang-format on

class GPTOSSExpert : public ov::pass::MatcherPass {
public:
    MOE_EXPERT_STATIC_INFO(GPTOSSExpert)
    GPTOSSExpert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class GPTOSSRouter : public ov::pass::MatcherPass {
public:
    MOE_ROUTER_STATIC_INFO(GPTOSSRouter)
    GPTOSSRouter(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Qwen3Expert : public ov::pass::MatcherPass {
public:
    MOE_EXPERT_STATIC_INFO(Qwen3Expert)
    Qwen3Expert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Qwen3Router : public ov::pass::MatcherPass {
public:
    MOE_ROUTER_STATIC_INFO(Qwen3Router)
    Qwen3Router(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Gemma4Expert : public ov::pass::MatcherPass {
public:
    MOE_EXPERT_STATIC_INFO(Gemma4Expert)
    Gemma4Expert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class Gemma4Router : public ov::pass::MatcherPass {
public:
    MOE_ROUTER_STATIC_INFO(Gemma4Router)
    Gemma4Router(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace moe
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
