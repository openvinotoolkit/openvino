// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/cache.hpp"
#include "matchers/subgraph/manager.hpp"
#include "matchers/subgraph/fused_names.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"
#include "matchers/subgraph/read_value_assign.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class GraphCache : public ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model,
                      const std::string& model_meta_data,
                      bool extract_body,
                      bool from_cache = false) override;
    void serialize_cache() override;

    static std::shared_ptr<GraphCache>& get(const std::string& device = "") {
        if (m_cache_instance == nullptr) {
            m_cache_instance = std::shared_ptr<GraphCache>(new GraphCache(device));
        }
        return m_cache_instance;
    }

    static void reset() {
        m_cache_instance.reset();
        m_cache_instance = nullptr;
    }

    void reset_cache() override {
        m_graph_cache.clear();
        reset();
    };

protected:
    std::map<std::shared_ptr<ov::Model>, ov::conformance::MetaInfo> m_graph_cache;
    // cache byte size
    uint64_t m_graph_cache_bytesize = 0;
    ExtractorsManager m_manager;
    ov::util::ModelComparator::Ptr m_model_comparator = ov::util::ModelComparator::get();
    std::shared_ptr<ov::Model> model_to_update = nullptr;
    static std::shared_ptr<GraphCache> m_cache_instance;

    GraphCache(const std::string& device = "") {
        ExtractorsManager::ExtractorsMap matchers = {
            { "repeat_pattern", RepeatPatternExtractor::Ptr(new RepeatPatternExtractor) },
            { "read_value_assign", ReadValueAssignExtractor::Ptr(new ReadValueAssignExtractor) },
        };
        try {
            matchers.insert({ "fused_names", FusedNamesExtractor::Ptr(new FusedNamesExtractor(device)) });
        } catch(const std::exception& e) {
            std::cout << "[ GRAPH CACHE ][ WARNING ] Fused names extractor is disabled according: " << e.what() << std::endl;
        }
        m_manager.set_extractors(matchers);
        m_cache_subdir = "subgraph";
    }

    void update_cache(const std::shared_ptr<ov::Model>& model,
                      const std::string& model_path,
                      const std::map<std::string, ov::conformance::InputInfo>& input_info,
                      const std::string& extractor_name,
                      size_t model_op_cnt);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov