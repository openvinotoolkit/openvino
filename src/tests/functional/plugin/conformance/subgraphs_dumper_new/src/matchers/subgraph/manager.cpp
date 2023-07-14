// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        auto extracted_patterns = it.second->extract(model, is_extract_body);
        result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());
    }
    return result;
}
