// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/op_types.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/graph_comparator.hpp"

#include "cache/graph_cache.hpp"
#include "utils/node.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

std::shared_ptr<GraphCache> GraphCache::m_cache_instance = nullptr;

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& model, const std::string& model_meta_data, bool extract_body) {
    auto model_total_op = model->get_ops().size() - model->get_output_size() - model->inputs().size();
    auto extracted_patterns = m_manager.run_extractors(model);
    if (extracted_patterns.empty()) {
        return;
    }
    for (const auto& cached_pattern : m_graph_cache) {
        auto it = extracted_patterns.begin();
        while (it != extracted_patterns.end()) {
            if (m_manager.match(cached_pattern.first, it->first)) {
                break;
            }
            ++it;
        }
        if (it == extracted_patterns.end()) {
            continue;
        }
        auto cached_model_size = cached_pattern.first->get_graph_size();
        auto pattern_model_size = it->first->get_graph_size();
        if (pattern_model_size < cached_model_size) {
            auto meta = cached_pattern.second;
            meta.update(model_meta_data, it->second, model_total_op);
            m_graph_cache.erase(cached_pattern.first);
            m_graph_cache.insert({it->first, meta});
        } else {
            m_graph_cache[cached_pattern.first].update(model_meta_data, it->second, model_total_op);
        }
        extracted_patterns.erase(it);
    }

    for (const auto& extracted_pattern : extracted_patterns) {
        auto meta = MetaInfo(model_meta_data, extracted_pattern.second, model_total_op);
        m_graph_cache.insert({model, meta});
    }
    return;
}

void GraphCache::serialize_cache() {
    for (const auto& cache_item : m_graph_cache) {
        serialize_model(cache_item, "subgraph");
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov