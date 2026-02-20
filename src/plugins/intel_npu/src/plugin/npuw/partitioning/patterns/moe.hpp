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

class GPTOSSExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::GPTOSSExpert");
    GPTOSSExpert(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

class GPTOSSRouter : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::moe::GPTOSSRouter");
    GPTOSSRouter(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag);
};

}  // namespace moe
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
