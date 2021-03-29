// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/matchers_manager.hpp"

using namespace SubgraphsDumper;

bool MatchersManager::match_any(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) {
    const auto matches = this->run_matchers(node, ref);
    return std::any_of(matches.begin(), matches.end(), [](bool i) { return i; });
}

bool MatchersManager::match_all(const std::shared_ptr<ngraph::Node> &node, const  std::shared_ptr<ngraph::Node> &ref) {
    const auto matches = this->run_matchers(node, ref);
    return std::all_of(matches.begin(), matches.end(), [](bool i) { return i; });
}

MatchersManager::MatchersManager(const std::string &cfg_path) {
    if (!cfg_path.empty()) {
//        m_cfg.load_file(cfg_path.c_str());
    }
    for (const auto &it : m_registry) {
        m_matchers[it.first] = it.second();
    }
}

std::vector<bool> MatchersManager::run_matchers(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) {
    std::vector<bool> matches;
    for (const auto &it : m_matchers) {
        matches.push_back(it.second->match(node, ref));
    }
    return matches;
}
