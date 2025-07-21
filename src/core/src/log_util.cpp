// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log_util.hpp"

#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/true.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/env_util.hpp"
#include "transformations/utils/gen_pattern.hpp"

namespace ov {
namespace util {

#ifdef ENABLE_OPENVINO_DEBUG
// Switch on verbose matching logging using OV_VERBOSE_LOGGING=true
static const bool verbose = ov::util::getenv_bool("OV_VERBOSE_LOGGING");

// These functions are used for printing nodes in a pretty way for matching logging
static std::string wrapped_type_str(const ov::gen_pattern::detail::GenericPattern& gp, bool verbose) {
    auto wrapped_type_info = gp.get_wrapped_type();
    auto version = wrapped_type_info.version_id;
    std::string res = "<";
    if (verbose && version)
        res += version + std::string("::");

    res += wrapped_type_info.name + std::string(">");
    return res;
}

static std::string wrapped_type_str(const ov::pass::pattern::op::WrapType& wt, bool verbose) {
    bool first = true;
    std::string res = "<";
    for (const auto& type : wt.get_wrapped_types()) {
        auto version = type.version_id;
        res += std::string(first ? "" : ", ");
        if (verbose && version)
            res += version + std::string("::");

        res += type.name;
        first = false;
    }
    res += ">";
    return res;
}

bool is_label_with_any_input(const ov::Node& node) {
    if (auto label = ov::as_type<const ov::pass::pattern::op::Label>(&node)) {
        if (label->get_input_size() == 1) {
            if (ov::as_type<const ov::pass::pattern::op::True>(label->input_value(0).get_node())) {
                return true;
            }
        }
    }
    return false;
}

bool true_any_input(const Output<Node>& output) {
    if (ov::as_type_ptr<ov::pass::pattern::op::True>(output.get_node_shared_ptr())) {
        if (output.get_target_inputs().size() == 1) {
            auto consumer_node = output.get_target_inputs().begin()->get_node();
            if (auto label = ov::as_type_ptr<ov::pass::pattern::op::Label>(consumer_node->shared_from_this())) {
                return true;
            }
        }
    }

    return false;
}

bool is_verbose_logging() {
    return verbose;
}

static std::string arguments_str(const OutputVector& input_values, bool verbose) {
    std::string sep = "";
    std::stringstream stream;
    stream << "(";

    for (const auto& arg : input_values) {
        if (verbose) {
            stream << sep << arg;
        } else {
            if (is_label_with_any_input(*arg.get_node())) {
                stream << sep << "any_input";
            } else {
                stream << sep << arg.get_node_shared_ptr()->get_type_name();
            }
        }
        sep = ", ";
    }
    stream << ")";

    return stream.str();
}

std::string node_version_type_name_str(const ov::Node& node) {
    return ov::util::node_version_type_str(node) + " " + node.get_name();
}

std::string node_version_type_str(const ov::Node& node) {
    std::string res;
    if (!verbose && is_label_with_any_input(node)) {
        return "any_input";
    }

    auto version = node.get_type_info().version_id;
    if (verbose && version)
        res = version + std::string("::");

    res += node.get_type_info().name;

    if (auto wrap_type = ov::as_type<const ov::pass::pattern::op::WrapType>(&node)) {
        res += wrapped_type_str(*wrap_type, verbose);
    } else if (auto generic_pattern = ov::as_type<const ov::gen_pattern::detail::GenericPattern>(&node)) {
        res += wrapped_type_str(*generic_pattern, verbose);
    }

    return res;
}

std::string node_with_arguments(const ov::Node& node) {
    std::string res;

    if (!verbose && is_label_with_any_input(node)) {
        return "any_input()";
    }

    auto version = node.get_type_info().version_id;
    if (verbose && version)
        res += version + std::string("::");

    res += node.get_type_info().name;

    if (auto wrap_type = ov::as_type<const ov::pass::pattern::op::WrapType>(&node)) {
        res += wrapped_type_str(*wrap_type, verbose);
    } else if (auto generic_pattern = ov::as_type<const ov::gen_pattern::detail::GenericPattern>(&node)) {
        res += wrapped_type_str(*generic_pattern, verbose);
    }

    if (verbose)
        res += " " + node.get_friendly_name();

    res += arguments_str(node.input_values(), verbose);

    return res;
}

std::string attribute_str(const ov::Any& attribute) {
    std::stringstream ss;
    attribute.print(ss);
    return ss.str();
}

#endif
}  // namespace util
}  // namespace ov
