// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/pass/pattern/multi_matcher.hpp"

using namespace ov::pass;
using namespace ov::pass::pattern;

MultiMatcher::MultiMatcher(const std::string& name) : m_name(name) {}

void MultiMatcher::register_patterns(const ov::NodeVector& patterns, Callback callback, bool strict) {
    m_callback = std::move(callback);
    m_patterns.clear();
    m_all_roots.clear();

    for (const auto& p : patterns) {
        m_patterns.push_back(PatternEntry{p->output(0), p, strict});
    }
}

bool MultiMatcher::run_on_model(const std::shared_ptr<Model>& model) {
    bool changed = false;
    m_matched_nodes.clear();

    std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>> matches_by_pattern;
    for (const auto& node : model->get_ordered_ops()) {
        for (const auto& pattern : m_patterns) {
            Matcher matcher(pattern.pattern, m_name, pattern.strict_mode);
            if (!matcher.match(node->output(0)))
                continue;

            m_all_roots.insert(node.get());

            const auto& match_map = matcher.get_pattern_value_map();
            const auto& matched_nodes = matcher.get_matched_nodes();

            for (const auto& n : matched_nodes)
                m_matched_nodes.insert(n.get());

            matches_by_pattern[pattern.root_ptr].push_back(match_map);
            break;  // Skip trying other patterns on this node
        }
    }

    if (!matches_by_pattern.empty()) {
        m_callback(matches_by_pattern);
        changed = true;
    }

    return changed;
}
