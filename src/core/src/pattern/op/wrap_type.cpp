// Copyright (C) 2018-2025 Intel Corporation
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
    // Check output indices when pattern explicitly expects multiple outputs
    // (i.e., when set_wrapped_output_size() was called with size > 1)
    // This ensures patterns like split->output(1) only match the correct output
    // while maintaining backward compatibility for existing patterns
    if (m_explicit_output_size > 1 && graph_value.get_node_shared_ptr()->get_output_size() > 1 &&
        pattern_value.get_index() != graph_value.get_index()) {
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
    pattern_map[shared_from_this()] = graph_value;
    matcher->add_node(graph_value);
    OPENVINO_LOG_WRAPTYPE3(matcher, get_input_size());
    auto res =
        (get_input_size() == 0 ? true
                               : matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr()));
    OPENVINO_LOG_WRAPTYPE4(matcher, res, get_input_size());
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

std::shared_ptr<ov::Node> ov::pass::pattern::wrap_const() {
    return ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
}