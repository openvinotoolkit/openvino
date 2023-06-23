// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_matchers/manager.hpp"

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

bool MatchersManager::match_any(const std::shared_ptr<ov::Node> &node,
                                const std::shared_ptr<ov::Node> &ref) {
    for (const auto &it : m_matchers) {
        if (it.second->match(node, ref)) return true;
    }
    return false;
}

bool MatchersManager::match_all(const std::shared_ptr<ov::Node> &node,
                                const std::shared_ptr<ov::Node> &ref) {
    const auto matches = this->run_matchers(node, ref);
    return std::all_of(matches.begin(), matches.end(), [](bool i) { return i; });
}

std::vector<bool> MatchersManager::run_matchers(const std::shared_ptr<ov::Node> &node,
                                                const std::shared_ptr<ov::Node> &ref) {
    std::vector<bool> matches;
    for (const auto &it : m_matchers) {
        matches.push_back(it.second->match(node, ref));
    }
    return matches;
}
