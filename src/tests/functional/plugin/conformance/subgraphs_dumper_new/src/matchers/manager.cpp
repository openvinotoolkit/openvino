// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/manager.hpp"

using namespace ov::tools::subgraph_dumper;

iMatcherConfig::Ptr MatchersManager::get_config(const std::shared_ptr<ov::Node> &node) const {
    if (node == nullptr) return nullptr;
    for (const auto &it : m_matchers) {
        auto default_config = it.second->get_config(node);
        if (default_config->op_in_config(node)) {
            return default_config;
        }
    }
    return nullptr;
}

bool MatchersManager::match(const std::shared_ptr<ov::Node> &node,
                            const std::shared_ptr<ov::Node> &ref) {
    for (const auto &it : m_matchers) {
        if (it.second->match(node, ref)) {
            return true;
        }
    }
    return false;
}

bool MatchersManager::match(const std::shared_ptr<ov::Model> &model,
                            const std::shared_ptr<ov::Model> &ref_model) {
    for (const auto &it : m_matchers) {
        if (it.second->match(model, ref_model)) {
            return true;
        }
    }
    return false;
}

std::list<ExtractedPattern>
MatchersManager::run_extractors(const std::shared_ptr<ov::Model> &model) {
    std::list<ExtractedPattern> result;
    for (const auto &it : m_matchers) {
        auto extracted_patterns = it.second->extract(model);
        result.insert(result.end(), extracted_patterns.begin(), extracted_patterns.end());
    }
    return result;
}
