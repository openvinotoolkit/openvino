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

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& model,
                              const std::string& model_meta_data,
                              bool extract_body) {
    std::cout << "[ INFO ][ GRAPH CACHE ] Processing model: " << model_meta_data << std::endl;
    auto model_total_op = model->get_ops().size() - model->get_output_size() - model->inputs().size();
    auto extracted_patterns = m_manager.extract(model, extract_body);
    if (extracted_patterns.empty()) {
        return;
    }
    while (!extracted_patterns.empty()) {
        auto it = extracted_patterns.begin();
        update_cache(it->first, model_meta_data, it->second, model_total_op);
        extracted_patterns.pop_front();
    }
    return;
}

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& extracted_model, const std::string& model_path,
                              const std::map<std::string, InputInfo>& input_info, size_t model_op_cnt) {
    std::shared_ptr<ov::Model> model_to_update = nullptr;
    for (const auto& cached_model : m_graph_cache) {
        if (m_manager.match(cached_model.first, extracted_model)) {
            model_to_update = cached_model.first;
            break;
        }
    }
    if (model_to_update == nullptr) {
        auto meta = MetaInfo(model_path, input_info, model_op_cnt);
        m_graph_cache.insert({ extracted_model, meta });
        return;
    }
    m_graph_cache[model_to_update].update(model_path, input_info, model_op_cnt);
    auto cached_model_size = model_to_update->get_graph_size();
    auto pattern_model_size = extracted_model->get_graph_size();
    if (pattern_model_size < cached_model_size) {
        auto meta = m_graph_cache[model_to_update];
        m_graph_cache.erase(model_to_update);
        m_graph_cache.insert({extracted_model, meta});
    }
}

void GraphCache::serialize_cache() {
    for (const auto& cache_item : m_graph_cache) {
        serialize_model(cache_item, "subgraph");
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov