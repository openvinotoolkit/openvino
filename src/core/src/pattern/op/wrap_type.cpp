// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/core/except.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

#ifdef ENABLE_OPENVINO_DEBUG
using namespace ov::util;
#endif

bool ov::pass::pattern::op::WrapType::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    if (std::none_of(m_wrapped_types.begin(), m_wrapped_types.end(),
                    [&](const NodeTypeInfo& type_info) {
                        return graph_value.get_node_shared_ptr()->get_type_info().is_castable(type_info);
                    })) {
        OV_LOG_MATCHING(matcher, level_string(matcher->level), "}  ", OV_RED, "NODES' TYPE DIDN'T MATCH. EXPECTED: ", ov::node_version_type_str(pattern_value.get_node_shared_ptr()),
                                                                                               ". OBSERVED: ", ov::node_version_type_str(graph_value.get_node_shared_ptr()));
        return false;
    }

    if (!m_predicate(graph_value)) {
        OV_LOG_MATCHING(matcher, level_string(matcher->level), "}  ", OV_RED, "NODES' TYPE MATCHED, but PREDICATE FAILED");
        return false;
    }

    auto& pattern_map = matcher->get_pattern_value_map();
    pattern_map[shared_from_this()] = graph_value;
    matcher->add_node(graph_value);
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "├─ ", OV_GREEN, "NODES' TYPE and PREDICATE MATCHED. CHECKING ", get_input_size(), " PATTERN ARGUMENTS: ");
    auto res =  (get_input_size() == 0
                ? true
                : matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr()));
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "│");
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "}  ", (res ? OV_GREEN : OV_RED), (res ? "ALL ARGUMENTS MATCHED" : "ARGUMENTS DIDN'T MATCH"));
    return res;
}

ov::NodeTypeInfo ov::pass::pattern::op::WrapType::get_wrapped_type() const {
    if (m_wrapped_types.size() > 1) {
        OPENVINO_THROW("get_wrapped_type() called on WrapType with more than one type");
    }
    return m_wrapped_types.at(0);
}

const std::vector<ov::NodeTypeInfo>& ov::pass::pattern::op::WrapType::get_wrapped_types() const {
    return m_wrapped_types;
}

std::ostream& ov::pass::pattern::op::WrapType::write_type_description(std::ostream& out) const {
    bool first = true;
    out << (m_wrapped_types.size() > 1 ? "<" : "");
    for (const auto& type : m_wrapped_types) {
        auto version = type.version_id;
        if (version)
            out << (first ? "" : ", ") << version << "::" << type.name;
        else
            out << (first ? "" : ", ") << type.name;
        first = false;
    }
    out << (m_wrapped_types.size() > 1 ? ">" : "");
    return out;
}

#ifdef ENABLE_OPENVINO_DEBUG
std::string ov::pass::pattern::op::WrapType::type_description_str(bool verbose) const {
    bool first = true;
    std::string res = "<";
    for (const auto& type : m_wrapped_types) {
        auto version = type.version_id;
        res += std::string(first ? "" : ", ");
        if (verbose)
            if (version)
                res += version + std::string("::");
        res += type.name;
        first = false;
    }
    res += ">";
    return res;
}
#endif