// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/cache.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "matchers/subgraph/fused_names.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class GraphCache final : public virtual ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data,
                      bool extract_body = true) override;
    void serialize_cache() override;

    static std::shared_ptr<GraphCache>& get() {
        if (m_cache_instance == nullptr) {
            m_cache_instance = std::shared_ptr<GraphCache>(new GraphCache);
        }
        return m_cache_instance;
    }

    static void reset() {
        m_cache_instance.reset();
        m_cache_instance = nullptr;
    }

private:
    std::map<std::shared_ptr<ov::Model>, MetaInfo> m_graph_cache;
    static std::shared_ptr<GraphCache> m_cache_instance;

    GraphCache() {
        MatchersManager::MatchersMap matchers = {
            { "fused_names", FusedNamesMatcher::Ptr(new FusedNamesMatcher) },
            { "repeat_patter", RepeatPatternMatcher::Ptr(new RepeatPatternMatcher) },
        };
        m_manager.set_matchers(matchers);
    }
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov