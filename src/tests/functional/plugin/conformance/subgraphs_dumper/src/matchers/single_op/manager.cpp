// Copyright (C) 2018-2025 Intel Corporation
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

void MatchersManager::set_shape_strict_match(bool shape_strict_match) {
    for (const auto& matcher : m_matchers) {
        matcher.second->set_strict_shape_match(shape_strict_match);
    }
}

void MatchersManager::set_match_attributes(bool match_attribute) {
    for (const auto& matcher : m_matchers) {
        matcher.second->set_match_attrib(match_attribute);
    }
}

void MatchersManager::set_match_in_types(bool match_in_types) {
    for (const auto& matcher : m_matchers) {
        matcher.second->set_match_in_types(match_in_types);
    }
}

bool MatchersManager::match(const std::shared_ptr<ov::Node> &node,
                            const std::shared_ptr<ov::Node> &ref) const {
    for (const auto& it : m_matchers) {
        if (it.second->match(node, ref)) {
            return true;
        }
    }
    return false;
}
