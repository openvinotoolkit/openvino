// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <chrono>
#include "matchers/subgraph/manager.hpp"

using namespace ov::tools::subgraph_dumper;

bool ExtractorsManager::match(const std::shared_ptr<ov::Model> &model,
                              const std::shared_ptr<ov::Model> &ref) {
    for (const auto &it : m_extractors) {
        if (it.second->match(model, ref)) {
            return true;
        }
    }
    return false;
}

std::list<ExtractedPattern>
ExtractorsManager::extract(const std::shared_ptr<ov::Model> &model, bool is_extract_body) {
    std::list<ExtractedPattern> result;
    for (const auto &it : m_extractors) {
        auto start = std::chrono::high_resolution_clock::now();
        it.second->set_extractor_name(it.first);
        auto extracted_patterns = it.second->extract(model, is_extract_body);
        result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());
        auto end = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[ INFO ][ EXTRACTOR DURATION ] " << it.first << " " << delta << "ms" << std::endl;
    }
    return result;
}
