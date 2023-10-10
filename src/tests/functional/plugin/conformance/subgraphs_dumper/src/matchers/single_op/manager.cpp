// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op/manager.hpp"

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
                            const std::shared_ptr<ov::Node> &ref) const {
    for (const auto &it : m_matchers) {
        if (it.second->match(node, ref)) {
            return true;
        }
    }
    return false;
}
