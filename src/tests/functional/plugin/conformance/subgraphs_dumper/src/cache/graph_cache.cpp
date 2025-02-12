// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"
#include "cache/graph_cache.hpp"
#include "utils/model.hpp"
#include "utils/cache.hpp"
#include "op_conformance_utils/utils/file.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

std::shared_ptr<GraphCache> GraphCache::m_cache_instance = nullptr;

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& model,
                              const std::string& model_meta_data,
                              bool extract_body,
                              bool from_cache) {
    std::cout << "[ INFO ][ GRAPH CACHE ] Processing model: " << model_meta_data << std::endl;
    auto model_total_op = model->get_ops().size() - model->get_output_size() - model->inputs().size();
    if (from_cache) {
        auto meta_path = ov::util::replace_extension(model_meta_data, "meta");
        auto meta = ov::conformance::MetaInfo::read_meta_from_file(meta_path);
        m_graph_cache.insert({ model, meta });
        m_graph_cache_bytesize += model->get_graph_size();
    } else {
        // const won't be cloned in case model takes > 50% RAM
        auto model_bytesize = model->get_graph_size();
        // check that Free RAM memory is enough. Serialize in other case
        // serialize graph cache in case graph cache bytesize > 4GB to avoid long search the same graphs
        if (m_graph_cache_bytesize + 2 * model_bytesize >= mem_size) {
            std::cout << "[ GRAPH CACHE ][ WARNING ] There are not enought RAM memory! Serialize graph cache" << std::endl;
            serialize_cache();
            m_graph_cache_bytesize = 0;
        }
        if (m_graph_cache_bytesize * 4 >= mem_size) {
            std::cout << "[ GRAPH CACHE ][ WARNING ] 25% of RAM is used by cache! Serialize graph cache" << std::endl;
            serialize_cache();
            m_graph_cache_bytesize = 0;
        }
        auto is_large_model = is_model_large_to_store_const(model);
        if (is_large_model) {
            auto model_bytesize_gb = model_bytesize;
            model_bytesize_gb >>= 30;
            auto mem_size_gb = mem_size;
            mem_size_gb >>= 30;
            std::cout << "[ GRAPH CACHE ][ WARNING ] Model  bytesize is " << model_bytesize_gb <<
            "GB. It is larger than 25% RAM size or >= 8GB: " << mem_size_gb << ". Constants won't be copied!" << std::endl;
        }
        auto extracted_patterns = m_manager.extract(model, extract_body, !is_large_model);
        if (extracted_patterns.empty()) {
            return;
        }
        while (!extracted_patterns.empty()) {
            auto it = *extracted_patterns.rbegin();
            update_cache(std::get<0>(it), model_meta_data, std::get<1>(it), std::get<2>(it), model_total_op);
            extracted_patterns.pop_back();
        }
    }
}

void GraphCache::update_cache(const std::shared_ptr<ov::Model>& extracted_model,
                              const std::string& model_path,
                              const std::map<std::string, ov::conformance::InputInfo>& input_info,
                              const std::string& extractor_name,
                              size_t model_op_cnt) {
    auto graph_name = extracted_model->get_friendly_name();
    auto this_op_cnt = extracted_model->get_ops().size() -
        extracted_model->get_parameters().size() - extracted_model->get_results().size();
    std::map<std::string, ov::conformance::InputInfo> updated_input_info;
    if (!m_graph_cache.empty() && model_to_update != nullptr) {
        auto comparator_res = m_model_comparator->match(extracted_model, model_to_update,
                                                        input_info, m_graph_cache.at(model_to_update).get_input_info());
        if (comparator_res.first) {
            updated_input_info = comparator_res.second;
        } else {
            model_to_update = nullptr;
        }
    }

    if (model_to_update == nullptr) {
        std::string serialized_model_path = "";
        for (const auto& extractor : m_manager.get_extractors()) {
            auto tmp_serialized_model_path = ov::util::path_join({ m_serialization_dir, m_cache_subdir, extractor.first, graph_name + ".xml" }).string();
            if (ov::util::file_exists(tmp_serialized_model_path)) {
                serialized_model_path = tmp_serialized_model_path;
                break;
            }
        }
        // if cached model was serialized
        if (!serialized_model_path.empty()) {
            // std::cout << "[ GRAPH CACHE ][ INFO ] Reading cached model: " << serialized_model_path << std::endl;
            auto bin_path = ov::util::replace_extension(serialized_model_path, ".bin");
            auto meta_path = ov::util::replace_extension(serialized_model_path, ".meta");
            auto cached_model = ov::util::core->read_model(serialized_model_path);
            auto cached_meta = ov::conformance::MetaInfo::read_meta_from_file(meta_path);

            ov::util::remove_path(serialized_model_path);
            ov::util::remove_path(bin_path);
            ov::util::remove_path(meta_path);

            m_graph_cache.insert({ cached_model, cached_meta });
            m_graph_cache_bytesize += cached_model->get_graph_size();

            auto comparator_res = m_model_comparator->match(extracted_model, cached_model,
                                                            input_info, cached_meta.get_input_info());
            if (comparator_res.first) {
                model_to_update = cached_model;
                updated_input_info = comparator_res.second;
            }
        } else {
            for (const auto& cached_model : m_graph_cache) {
                auto comparator_res = m_model_comparator->match(extracted_model, cached_model.first,
                                                                input_info, cached_model.second.get_input_info());
                if (comparator_res.first) {
                    model_to_update = cached_model.first;
                    updated_input_info = comparator_res.second;
                    break;
                } else {
                    auto is_subgraph = m_model_comparator->is_subgraph(extracted_model, cached_model.first,
                                                                       input_info, cached_model.second.get_input_info());
                    // in case if one model is subgraph of other to update model meta info and remove subgraph from cache
                    if (std::get<0>(is_subgraph)) {
                        std::shared_ptr<ov::Model> graph, subgraph;
                        std::map<std::string, ov::conformance::InputInfo> graph_in_info, subgraph_in_info;
                        std::tie(std::ignore, subgraph, graph, subgraph_in_info, graph_in_info) = is_subgraph;
                        if (subgraph == cached_model.first) {
                            auto meta = m_graph_cache[subgraph];
                            meta.set_input_info(graph_in_info);
                            m_graph_cache_bytesize += (graph->get_graph_size() - subgraph->get_graph_size());
                            m_graph_cache.erase(subgraph);
                            m_graph_cache.insert({graph, meta});
                        } else {
                            m_graph_cache[cached_model.first].update(model_path,
                                                                     subgraph_in_info,
                                                                     model_op_cnt,
                                                                     this_op_cnt,
                                                                     extractor_name);
                        }
                        return;
                    } else {
                        auto matched_ops = std::get<3>(m_model_comparator->is_subgraph(extracted_model, cached_model.first));
                        auto cached_model_op_cnt =
                            cached_model.first->get_ops().size() - cached_model.second.get_input_info().size() -
                            cached_model.first->get_results().size();
                        auto extracted_model_op_cnt =
                            extracted_model->get_ops().size() - input_info.size() - extracted_model->get_results().size();

                        bool is_same_paired_op_cnt = ov::util::is_same_paired_op_cnt(extracted_model, cached_model.first);
                        if (matched_ops.size() > 0.75 * extracted_model_op_cnt && is_same_paired_op_cnt) {
                            if (cached_model_op_cnt > extracted_model_op_cnt) {
                                return;
                            }
                            m_graph_cache_bytesize += (extracted_model->get_graph_size() - cached_model.first->get_graph_size());
                            m_graph_cache.erase(cached_model.first);
                            ov::conformance::MetaInfo meta(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
                            m_graph_cache.insert({extracted_model, meta});
                            return;
                        }
                    }
                }
            }
        }
    }

    if (model_to_update == nullptr) {
        ov::conformance::MetaInfo meta(model_path, input_info, model_op_cnt, this_op_cnt, extractor_name);
        model_to_update = extracted_model;
        m_graph_cache.insert({ model_to_update, meta });
        m_graph_cache_bytesize += extracted_model->get_graph_size();
        return;
    }
    m_graph_cache[model_to_update].update(model_path, updated_input_info, model_op_cnt, this_op_cnt, extractor_name);
    auto cached_model_size = model_to_update->get_graph_size();
    auto pattern_model_size = extracted_model->get_graph_size();
    if (pattern_model_size < cached_model_size) {
        m_graph_cache_bytesize -= (cached_model_size - pattern_model_size);
        auto meta = m_graph_cache[model_to_update];
        auto matched_ops = m_model_comparator->get_matched_ops_in_graphs(model_to_update, extracted_model);
        auto new_in_info = ov::util::align_input_info(model_to_update, extracted_model,
                                                      m_graph_cache.at(model_to_update).get_input_info(), input_info,
                                                      matched_ops);
        meta.set_input_info(new_in_info);
        m_graph_cache.erase(model_to_update);
        model_to_update = extracted_model;
        m_graph_cache.insert({model_to_update, meta});
    }
}

void GraphCache::serialize_cache() {
    while (!m_graph_cache.empty()) {
        auto cache_item = m_graph_cache.begin();
        auto rel_dir = ov::util::path_join({ m_cache_subdir,
                                             ov::util::get_model_type(cache_item->first),
                                             cache_item->second.get_any_extractor() }).string();
        serialize_model(*cache_item, rel_dir);
        m_graph_cache.erase(cache_item);
    }
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
