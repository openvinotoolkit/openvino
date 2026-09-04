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
