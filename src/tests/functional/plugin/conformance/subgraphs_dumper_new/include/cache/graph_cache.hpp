// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/cache.hpp"
#include "cache/meta/input_info.hpp"
#include "matchers/subgraph/manager.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "matchers/subgraph/fused_names.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class GraphCache : public ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model,
                      const std::string& model_meta_data,
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

    void reset_cache() override {
        reset();
    };

protected:
    std::map<std::shared_ptr<ov::Model>, MetaInfo> m_graph_cache;
    ExtractorsManager m_manager = ExtractorsManager();
    static std::shared_ptr<GraphCache> m_cache_instance;

    GraphCache() {
        ExtractorsManager::ExtractorsMap matchers = {
            { "fused_names", FusedNamesExtractor::Ptr(new FusedNamesExtractor) },
            { "repeat_pattern", RepeatPatternExtractor::Ptr(new RepeatPatternExtractor) },
        };
        m_manager.set_extractors(matchers);
    }

    void update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_path,
                      const std::map<std::string, InputInfo>& input_info, size_t model_op_cnt);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov