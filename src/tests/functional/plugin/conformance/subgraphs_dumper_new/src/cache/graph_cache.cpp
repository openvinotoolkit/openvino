// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "openvino/op/util/op_types.hpp"
#include "openvino/util/file_util.hpp"

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
        auto it = *extracted_patterns.begin();
        update_cache(std::get<0>(it), model_meta_data, std::get<1>(it), std::get<2>(it), model_total_op);
        extracted_patterns.pop_front();
    }
    return;
}

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& extracted_model, const std::string& model_path,
                              std::map<std::string, InputInfo>& input_info, const std::string& extractor_name, size_t model_op_cnt) {
    std::shared_ptr<ov::Model> model_to_update = nullptr;
    for (const auto& cached_model : m_graph_cache) {
        if (m_manager.match(cached_model.first, extracted_model)) {
            model_to_update = cached_model.first;
            break;
        }
    }

    auto this_op_cnt = extracted_model->get_ops().size() -
        extracted_model->get_parameters().size() - extracted_model->get_results().size();
    if (model_to_update == nullptr) {
        auto meta = MetaInfo(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
        m_graph_cache.insert({ extracted_model, meta });
        return;
    } else {
        bool is_update_required = false;
        auto cached_in_info = m_graph_cache[model_to_update].get_input_info();
        for (const auto& in_info : cached_in_info) {
            if (!input_info.count(in_info.first)) {
                is_update_required = true;
                break;
            }
        }
        if (is_update_required) {
            // align matched model names
            auto cached_model_ops = model_to_update->get_ordered_ops();
            auto extracted_model_ops = extracted_model->get_ordered_ops();
            size_t ordered_ops_size = extracted_model_ops.size();
            if (ordered_ops_size != cached_model_ops.size()) {
                throw std::runtime_error("Matched models are different!");
            }
            for (size_t i = 0; i < ordered_ops_size; ++i) {
                if (extracted_model_ops[i]->get_type_info() != cached_model_ops[i]->get_type_info()) {
                    throw std::runtime_error("Matched models are different!");
                }
                auto extracted_op_name = extracted_model_ops[i]->get_friendly_name();
                auto cached_op_name = cached_model_ops[i]->get_friendly_name();
                if (cached_in_info.count(cached_op_name)) {
                    auto in_info = input_info[extracted_op_name];
                    input_info.erase(extracted_op_name);
                    input_info.insert({ cached_op_name, in_info });
                }
                extracted_model_ops[i]->set_friendly_name(cached_op_name);
            }
        }
    }
    m_graph_cache[model_to_update].update(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
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
        auto rel_dir = ov::util::path_join({ "subgraph", cache_item.second.get_any_extractor() });
        serialize_model(cache_item, rel_dir);
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov