// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

bool ov::pass::pattern::op::WrapType::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    if (m_strict_output_index && pattern_value.get_index() != graph_value.get_index()) {
        OPENVINO_LOG_WRAPTYPE2(matcher);
        return false;
    }

    if (std::none_of(m_wrapped_types.begin(), m_wrapped_types.end(), [&](const NodeTypeInfo& type_info) {
            return graph_value.get_node_shared_ptr()->get_type_info().is_castable(type_info);
        })) {
        OPENVINO_LOG_WRAPTYPE1(matcher, pattern_value, graph_value);
        return false;
    }

    if (!m_predicate(matcher, graph_value)) {
        OPENVINO_LOG_WRAPTYPE2(matcher);
        return false;
    }

    auto& pattern_map = matcher->get_pattern_value_map();
    // Opt-in: bind this node to a single physical producer across all edges. On a
    // repeat visit the node is already validated, so skip re-matching its arguments.
    if (m_strict_output_index) {
        auto it = pattern_map.find(shared_from_this());
        if (it != pattern_map.end()) {
            return it->second.get_node() == graph_value.get_node();
        }
    }
    pattern_map[shared_from_this()] = graph_value;
    matcher->add_node(graph_value);
    OPENVINO_LOG_WRAPTYPE3(matcher, get_input_size());
    auto res =
        (get_input_size() == 0 ? true
                               : matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr()));
    OPENVINO_LOG_WRAPTYPE4(matcher, res, get_input_size());
    return res;
}

void ov::pass::pattern::op::WrapType::on_output_access(size_t output_index) {
    if (m_strict_output_index && output_index >= get_output_size()) {
        set_output_size(output_index + 1);
    }
}

void ov::pass::pattern::op::WrapType::validate_output_index(size_t output_index) const {
    if (output_index > 0 && output_index >= get_output_size()) {
        OPENVINO_THROW("Output ",
                       output_index,
                       " is not available on WrapType, use ",
                       m_strict_output_index ? "the non-const output()" : "wrap_type_strict_index<T>()");
    }
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

std::shared_ptr<ov::Node> ov::pass::pattern::wrap_const() {
    return ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
}