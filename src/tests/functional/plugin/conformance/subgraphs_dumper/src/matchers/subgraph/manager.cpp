// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/manager.hpp"

#include <chrono>

using namespace ov::tools::subgraph_dumper;

std::vector<SubgraphExtractor::ExtractedPattern>
ExtractorsManager::extract(const std::shared_ptr<ov::Model> &model,
                           bool is_extract_body,
                           bool is_copy_constants) {
    std::vector<SubgraphExtractor::ExtractedPattern> result;
    for (const auto &it : m_extractors) {
        // extract patterns from original models
        auto start = std::chrono::high_resolution_clock::now();
        it.second->set_extractor_name(it.first);
        it.second->set_extract_body(is_extract_body);
        it.second->set_save_const(is_copy_constants);
        auto extracted_patterns = it.second->extract(model);
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
