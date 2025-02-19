// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log_util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/gen_pattern.hpp"

namespace ov {
namespace util {

#ifdef ENABLE_OPENVINO_DEBUG
// Switch on verbose matching logging using OV_VERBOSE_LOGGING=true
static const bool verbose = ov::util::getenv_bool("OV_VERBOSE_LOGGING");

// These functions are used for printing nodes in a pretty way for matching logging
std::string node_version_type_str(const std::shared_ptr<ov::Node>& node) {
    auto version = node->get_type_info().version_id;
    std::string res;
    if (verbose)
        if (version)
            res = version + std::string("::");
    res += node->get_type_info().name;

    if (auto wrap_type = ov::as_type_ptr<ov::pass::pattern::op::WrapType>(node)) {
        res += wrap_type->type_description_str(verbose);
    } else if (auto generic_pattern = ov::as_type_ptr<ov::gen_pattern::detail::GenericPattern>(node)) {
        res += generic_pattern->get_wraped_type_str(verbose);
    }

    return res;
}

std::string node_version_type_name_str(const std::shared_ptr<ov::Node>& node) {
    return ov::util::node_version_type_str(node) + std::string(" ") + node->get_name();
}

std::string node_with_arguments(const std::shared_ptr<ov::Node>& node) {
    std::string res;
    auto version = node->get_type_info().version_id;
    if (verbose)
        if (version)
            res += version + std::string("::");
    res += node->get_type_info().name;

    if (auto wrap_type = ov::as_type_ptr<ov::pass::pattern::op::WrapType>(node)) {
        res += wrap_type->type_description_str(verbose);
    } else if (auto generic_pattern = ov::as_type_ptr<ov::gen_pattern::detail::GenericPattern>(node)) {
        res += generic_pattern->get_wraped_type_str(verbose);
    }

    if (verbose)
        res += std::string(" ") + node->get_name();

    std::string sep = "";
    std::stringstream stream;
    stream << "(";
    for (const auto& arg : node->input_values()) {
        if (verbose)
            stream << sep << arg;
        else
            stream << sep << arg.get_node_shared_ptr()->get_type_name();
        sep = ", ";
    }
    stream << ")";

    res += stream.str();

    return res;
}

#endif
}
}