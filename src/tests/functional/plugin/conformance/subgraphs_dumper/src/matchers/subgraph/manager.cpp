// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "matchers/subgraph/manager.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &ref) {
    // `match` is not virtual method in base `SubgraphExtractor` class
    // we can use function from any `extractor` to avoid of cycle
    if (!m_extractors.empty()) {
        if (m_extractors.begin()->second->match(model, ref)) {
            return true;
        }
    }
    return false;
}

ExtractorsManager::ExtractedSubgraphTuple
ExtractorsManager::is_subgraph(const std::shared_ptr<ov::Model> &model,
                               const std::shared_ptr<ov::Model> &ref_model,
                               const std::map<std::string, InputInfo> &in_info,
                               const std::map<std::string, InputInfo> &in_info_ref) {
    if (!m_extractors.empty()) {
        // `is_subgraph` is not virtual method in base `SubgraphExtractor` class
        // we can use function from any `extractor` to avoid of cycle
        auto extractor_res = m_extractors.begin()->second->is_subgraph(model, ref_model);
        if (std::get<0>(extractor_res)) {
            std::map<std::string, InputInfo> graph_in_info, subgraph_in_info;
            if (std::get<1>(extractor_res) == model && std::get<2>(extractor_res) == ref_model) {
                graph_in_info = in_info;
                subgraph_in_info = in_info_ref;
            } else if (std::get<1>(extractor_res) == ref_model && std::get<2>(extractor_res) == model) {
                graph_in_info = in_info_ref;
                subgraph_in_info = in_info;
            } else {
                throw std::runtime_error("Generated models are incompatible with original ones!");
            }
            try {
                subgraph_in_info = align_input_info(std::get<2>(extractor_res), std::get<1>(extractor_res), subgraph_in_info, graph_in_info);
            } catch(std::exception) {
                return { false, nullptr, nullptr, {}, {} };
            }
            return { true, std::get<1>(extractor_res), std::get<2>(extractor_res), graph_in_info, subgraph_in_info };
        }
    }
    return { false, nullptr, nullptr, {}, {} };
}

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &ref,
                              std::map<std::string, InputInfo> &in_info,
                              const std::map<std::string, InputInfo> &in_info_ref) {
    if (match(model, ref)) {
        try {
            in_info = align_input_info(model, ref, in_info, in_info_ref);
            return true;
        } catch (std::exception) {
            return false;
        }
    }
    return false;
}

std::map<std::string, InputInfo>
ExtractorsManager::align_input_info(const std::shared_ptr<ov::Model>& model,
                                    const std::shared_ptr<ov::Model>& model_ref,
                                    const std::map<std::string, InputInfo>& in_info,
                                    const std::map<std::string, InputInfo>& in_info_ref,
                                    const std::map<std::string, std::string> &matched_op) {
    std::map<std::string, InputInfo> new_input_info = in_info;
    bool is_update_required = false;
    for (const auto& in_info_item : in_info_ref) {
        if (!in_info.count(in_info_item.first)) {
            is_update_required = true;
            break;
        } else if (in_info.at(in_info_item.first).is_const != in_info_item.second.is_const) {
            throw std::runtime_error("Impossible to update input info!!!");
        }
    }
    if (is_update_required) {
        // align matched model names
        auto ref_model_ops = model_ref->get_ordered_ops();
        auto model_ops = model->get_ordered_ops();
        size_t ref_ordered_ops_size = ref_model_ops.size();
        size_t ordered_ops_size = model_ops.size();
        if (ref_ordered_ops_size != ordered_ops_size && matched_op.empty()) {
            throw std::runtime_error("Matched models can not be compared according different op numbers!");
        }
        for (size_t i = 0; i < ref_ordered_ops_size; ++i) {
            auto model_op_name = i < ordered_ops_size ? model_ops[i]->get_friendly_name() : "";
            auto model_ref_op_name = ref_model_ops[i]->get_friendly_name();
            if (!in_info_ref.count(model_ref_op_name) && !in_info.count(model_op_name)) {
                continue;
            }
            auto input_info = matched_op.empty() ? new_input_info[model_op_name] : in_info_ref.at(model_ref_op_name);
            std::string input_name = matched_op.count(model_ref_op_name) ? matched_op.at(model_ref_op_name) : model_op_name;
            if (new_input_info.count(input_name)) {
                if (input_info.is_const != in_info_ref.at(model_ref_op_name).is_const) {
                    throw std::runtime_error("Impossible to update input info!!!");
                }
                if (!matched_op.empty()) {
                    input_info = new_input_info.at(input_name);
                }
                new_input_info.erase(input_name);
            }
            new_input_info.insert({ model_ref_op_name, input_info });
        }
    }
    return new_input_info;
}

std::list<ExtractedPattern>
ExtractorsManager::extract(const std::shared_ptr<ov::Model> &model,
                           bool is_extract_body,
                           bool is_copy_constants) {
    std::list<ExtractedPattern> result;
    for (const auto &it : m_extractors) {
        // extract patterns from original models
        auto start = std::chrono::high_resolution_clock::now();
        it.second->set_extractor_name(it.first);
        auto extracted_patterns = it.second->extract(model, is_extract_body, is_copy_constants);
        result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());
        auto end = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[ INFO ][ EXTRACTOR DURATION ][ ORIGINAL MODEL ] " << it.first << " " << delta << "ms" << std::endl;

        // todo: enable it after validation
        // if (!is_dynamic_model(model)) {
        //     // extract patterns from models after `constant_folding` pass
        //     ov::pass::Manager manager;
        //     manager.register_pass<ov::pass::ConstantFolding>();
        //     manager.run_passes(model);
        //     extracted_patterns = it.second->extract(model, is_extract_body, is_copy_constants);
        //     result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());

        //     end = std::chrono::high_resolution_clock::now();
        //     delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        //     std::cout << "[ INFO ][ EXTRACTOR DURATION ][ CONSTANT FOLDING ] " << it.first << " " << delta << "ms" << std::endl;
        // }
    }
    return result;
}
