// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/matcher.hpp"

#include <algorithm>
#include <regex>

#include "openvino/op/util/op_types.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"

namespace ov {
bool is_used(Node* node);

namespace pass {
namespace pattern {
MatcherState::MatcherState(Matcher* matcher)
    : m_matcher(matcher),
      m_pattern_value_map(matcher->m_pattern_map),
      m_watermark(matcher->m_matched_list.size()),
      m_capture_size(matcher->m_pattern_value_maps.size()) {}

namespace {
Output<Node> make_node_output(const std::shared_ptr<Node>& node) {
    return node->get_output_size() == 1 ? node->output(0) : std::make_shared<op::AnyOutput>(node)->output(0);
}
}  // namespace

Matcher::Matcher(std::shared_ptr<Node> pattern_node) : m_pattern_node(make_node_output(pattern_node)) {}

Matcher::Matcher(std::shared_ptr<Node> pattern_node, const std::string& name)
    : m_pattern_node(make_node_output(pattern_node)),
      m_name(name) {}

Matcher::Matcher(std::shared_ptr<Node> pattern_node, const std::string& name, bool strict_mode)
    : Matcher(make_node_output(pattern_node), name, strict_mode) {}

MatcherState::~MatcherState() {
    if (m_restore) {
        if (!m_matcher->m_matched_list.empty()) {
            m_matcher->m_matched_list.erase(m_matcher->m_matched_list.begin() + m_watermark,
                                            m_matcher->m_matched_list.end());
        }

        if (!m_matcher->m_pattern_value_maps.empty()) {
            m_matcher->m_pattern_value_maps.erase(m_pattern_value_maps.begin() + m_capture_size,
                                                  m_pattern_value_maps.end());
        }

        m_matcher->m_pattern_map = m_pattern_value_map;
    }
}

bool MatcherState::finish(bool is_successful) {
    m_restore = !is_successful;
    return is_successful;
}
PatternMap Matcher::get_pattern_map() const {
    return as_pattern_map(m_pattern_map);
}
size_t Matcher::add_node(Output<Node> value) {
    size_t result = m_matched_list.size();
    m_matched_list.push_back(value);
    return result;
}

std::shared_ptr<Node> Matcher::get_match_root() {
    return m_match_root.get_node_shared_ptr();
}

MatcherState Matcher::start_match() {
    return MatcherState(this);
}
Output<Node> Matcher::get_match_value() {
    return m_match_root;
}
void Matcher::capture(const std::set<Node*>& static_nodes) {
    m_pattern_value_maps.push_back(m_pattern_map);
    m_pattern_map.clear();
    for (auto key_value : m_pattern_value_maps.back()) {
        if (static_nodes.count(key_value.first.get()) > 0) {
            m_pattern_map.insert(key_value);
        }
    }
}

namespace {
ov::NodeVector get_subgraph_outputs(const NodeVector& nodes, const NodeVector& exclusions, bool ignore_unused) {
    const std::set<std::shared_ptr<Node>> exclusions_set(exclusions.begin(), exclusions.end());
    const std::set<std::shared_ptr<Node>> nodes_set(nodes.begin(), nodes.end());

    NodeVector outputs;

    for (const auto& n : nodes) {
        if (exclusions_set.count(n) != 0)
            continue;

        for (const auto& u : n->get_users()) {
            bool add_output = nodes_set.count(u) == 0 && (!ignore_unused || is_used(u.get()));
            if (add_output) {
                outputs.push_back(n);
            }
        }
    }
    return outputs;
}
}  // namespace

bool Matcher::is_contained_match(const NodeVector& exclusions, bool ignore_unused) {
    if (exclusions.empty()) {
        NodeVector label_exclusions;
        for (const auto& entry : m_pattern_map) {
            // leaf label
            if (entry.first->get_input_size() == 0) {
                label_exclusions.push_back(entry.second.get_node_shared_ptr());
            }
        }
        return get_subgraph_outputs(get_matched_nodes(), label_exclusions, ignore_unused).size() < 2;
    }

    return get_subgraph_outputs(get_matched_nodes(), exclusions, false).size() < 2;
}

bool Matcher::match_value(const ov::Output<Node>& pattern_value, const ov::Output<Node>& graph_value) {
    std::shared_ptr<Node> pattern_node = pattern_value.get_node_shared_ptr();
    std::shared_ptr<Node> graph_node = graph_value.get_node_shared_ptr();

    return pattern_node->match_value(this, pattern_value, graph_value);
}

bool Matcher::match_permutation(const OutputVector& pattern_args, const OutputVector& args) {
    for (size_t i = 0; i < args.size(); i++) {
        if (!match_value(pattern_args.at(i), args.at(i))) {
            OPENVINO_DEBUG("[MATCHER] Aborting. Argument ",
                           i,
                           " (",
                           args.at(i).get_node()->get_friendly_name(),
                           ") mismatch");
            return false;
        }
    }
    return true;
}

bool Matcher::match_arguments(Node* pattern_node, const std::shared_ptr<Node>& graph_node) {
    OPENVINO_DEBUG("[MATCHER] Match arguments at");
    OPENVINO_DEBUG("\t", *graph_node);
    OPENVINO_DEBUG("for pattern");
    OPENVINO_DEBUG("\t", *pattern_node);

    auto args = graph_node->input_values();
    auto pattern_args = pattern_node->input_values();

    if (args.size() != pattern_args.size()) {
        OPENVINO_DEBUG("[MATCHER] Aborting. Args count mismatch: candidate: ",
                       args.size(),
                       ";  pattern: ",
                       pattern_args.size());
        return false;
    }

    if (ov::op::util::is_commutative(graph_node)) {
        // TODO: [nikolayk] we don't really have to use lexicographically-based perms,
        // heap's algo should be faster
        std::sort(begin(pattern_args),
                  end(pattern_args),
                  [](const ov::Output<ov::Node>& n1, const ov::Output<ov::Node>& n2) {
                      return n1 < n2;
                  });
        do {
            auto saved = start_match();
            if (match_permutation(pattern_args, args)) {
                return saved.finish(true);
            }
        } while (std::next_permutation(begin(pattern_args),
                                       end(pattern_args),
                                       [](const ov::Output<ov::Node>& n1, const ov::Output<ov::Node>& n2) {
                                           return n1 < n2;
                                       }));
    } else {
        return match_permutation(pattern_args, args);
    }

    OPENVINO_DEBUG("[MATCHER] Aborting");
    return false;
}

bool Matcher::match(const Output<Node>& graph_value) {
    return match(graph_value, PatternValueMap{});
}

bool Matcher::match(std::shared_ptr<Node> node) {
    return match(node->output(0));
}
bool Matcher::match(const Output<Node>& graph_value, const PatternValueMap& previous_matches) {
    clear_state();

    // insert previous matches
    m_pattern_map.insert(previous_matches.cbegin(), previous_matches.cend());
    auto saved = start_match();
    bool is_match = saved.finish(match_value(m_pattern_node, graph_value));
    if (is_match) {
        m_match_root = graph_value;
    }
    return is_match;
}

bool Matcher::match(const Output<Node>& graph_value, const PatternMap& previous_matches) {
    return match(graph_value, as_pattern_value_map(previous_matches));
}

void Matcher::clear_state() {
    m_match_root.reset();
    m_pattern_map.clear();
    m_pattern_value_maps.clear();
    m_matched_list.clear();
}
}  // namespace pattern
}  // namespace pass
}  // namespace ov
