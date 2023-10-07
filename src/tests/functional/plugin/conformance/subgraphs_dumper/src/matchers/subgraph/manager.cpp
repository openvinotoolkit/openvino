// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "matchers/subgraph/manager.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

ExtractorsManager::ExtractedSubgraphTuple
ExtractorsManager::is_subgraph(const std::shared_ptr<ov::Model> &model,
                               const std::shared_ptr<ov::Model> &ref_model,
                               const std::map<std::string, InputInfo> &in_info,
                               const std::map<std::string, InputInfo> &in_info_ref) {
    auto extractor_res = model_comparator->is_subgraph(model, ref_model);
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
            subgraph_in_info = model_comparator->align_input_info(std::get<2>(extractor_res), std::get<1>(extractor_res), subgraph_in_info, graph_in_info);
        } catch(std::exception) {
            return { false, nullptr, nullptr, {}, {} };
        }
        return { true, std::get<1>(extractor_res), std::get<2>(extractor_res), graph_in_info, subgraph_in_info };
    }
    return { false, nullptr, nullptr, {}, {} };
}

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &model_ref,
                              std::map<std::string, InputInfo> &in_info,
                              const std::map<std::string, InputInfo> &in_info_ref) {
    return model_comparator->match(model, model_ref, in_info, in_info_ref);
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
