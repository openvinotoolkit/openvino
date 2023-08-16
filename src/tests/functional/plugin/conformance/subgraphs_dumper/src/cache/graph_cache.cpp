// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "openvino/op/util/op_types.hpp"
#include "openvino/util/file_util.hpp"

#include "functional_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/file_utils.hpp"

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
    // todo: check the number 8GB
    if (m_graph_cache_bytesize >> 33 > 0) {
        std::cout << "[ GRAPH CACHE ][ WARNING ] Cache size > 8 GB. Serialize graph cache" << std::endl;
        serialize_cache();
        // m_graph_cache.clear();
        m_graph_cache_bytesize = 0;
    }

    auto graph_name = extracted_model->get_friendly_name();
    std::string serialized_model_path = "";
    for (const auto& extractor : m_manager.get_extractors()) {
        auto tmp_serialized_model_path = ov::util::path_join({ m_serialization_dir, "subgraph", extractor.first, graph_name + ".xml" });
        if (ov::util::file_exists(serialized_model_path)) {
            serialized_model_path = tmp_serialized_model_path;
            break;
        }
    }

    std::shared_ptr<ov::Model> model_to_update = nullptr;
    // if cached model was serialized
    if (!serialized_model_path.empty()) {
        std::cout << "[ GRAPH CACHE ][ INFO ] Reading cached model: " << serialized_model_path << std::endl;
        auto bin_path = ov::test::utils::replaceExt(serialized_model_path, ".bin");
        auto meta_path = ov::test::utils::replaceExt(serialized_model_path, ".meta");
        auto cached_model = ov::test::utils::PluginCache::get().core()->read_model(serialized_model_path);
        auto cached_meta = MetaInfo::read_meta_from_file(meta_path);

        ov::test::utils::removeFile(serialized_model_path);
        ov::test::utils::removeFile(bin_path);
        ov::test::utils::removeFile(meta_path);
        m_graph_cache.insert({ cached_model, cached_meta });
        m_graph_cache_bytesize += cached_model->get_graph_size();
        model_to_update = cached_model;
        input_info = m_manager.align_input_info(extracted_model, model_to_update,
                                                input_info, cached_meta.get_input_info());
    } else {
        for (const auto& cached_model : m_graph_cache) {
            if (m_manager.match(extracted_model, cached_model.first,
                                input_info, cached_model.second.get_input_info())) {
                model_to_update = cached_model.first;
                break;
            }
        }
    }

    auto this_op_cnt = extracted_model->get_ops().size() -
        extracted_model->get_parameters().size() - extracted_model->get_results().size();
    if (model_to_update == nullptr) {
        auto meta = MetaInfo(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
        m_graph_cache.insert({ extracted_model, meta });
        m_graph_cache_bytesize += extracted_model->get_graph_size();
        return;
    }
    m_graph_cache[model_to_update].update(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
    auto cached_model_size = model_to_update->get_graph_size();
    auto pattern_model_size = extracted_model->get_graph_size();
    if (pattern_model_size < cached_model_size) {
        m_graph_cache_bytesize -= (cached_model_size - pattern_model_size);
        auto meta = m_graph_cache[model_to_update];
        m_graph_cache.erase(model_to_update);
        m_graph_cache.insert({extracted_model, meta});
    }
}

void GraphCache::serialize_cache() {
    // for (const auto& cache_item : m_graph_cache) {
        auto it = m_graph_cache.begin();
        while (it != m_graph_cache.end()) {
            auto rel_dir = ov::util::path_join({ "subgraph", it->second.get_any_extractor() });
            serialize_model(*it, rel_dir);
            m_graph_cache.erase(it->first);
            it = m_graph_cache.begin();
        }
        auto a = 0;
    // }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
