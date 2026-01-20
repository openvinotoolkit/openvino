// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/unconverted_ops_report.hpp"

#include <sstream>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace frontend {
namespace {

std::optional<FrameworkNodeErrorInfo> extract_from_framework_node(
    const std::shared_ptr<ov::op::util::FrameworkNode>& fw_node) {
    if (!fw_node) {
        return std::nullopt;
    }
    FrameworkNodeErrorInfo info;
    const auto& attrs = fw_node->get_attrs();
    info.op_type = attrs.get_type_name();
    if (info.op_type.empty()) {
        info.op_type = fw_node->get_type_name();
    }
    if (info.op_type.empty()) {
        info.op_type = fw_node->get_friendly_name();
    }
    info.failure_message = "This is OpenVINO internal type.";
    return info;
}

std::optional<FrameworkNodeErrorInfo> extract_node_info(const std::shared_ptr<ov::Node>& node,
                                                        const std::vector<UnconvertedOpExtractor>& extractors) {
    if (!node) {
        return std::nullopt;
    }
    for (const auto& extractor : extractors) {
        if (extractor) {
            if (auto info = extractor(node)) {
                return info;
            }
        }
    }
    if (const auto& fw_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(node)) {
        return extract_from_framework_node(fw_node);
    }
    return std::nullopt;
}

void collect_from_model(const std::shared_ptr<ov::Model>& model,
                        const std::vector<UnconvertedOpExtractor>& extractors,
                        UnconvertedOpMap& report) {
    if (!model) {
        return;
    }
    for (const auto& node : model->get_ordered_ops()) {
        if (auto info = extract_node_info(node, extractors)) {
            if (!info->op_type.empty() && report.count(info->op_type) == 0) {
                report.emplace(info->op_type, info->failure_message);
            }
        }
        if (const auto& multigraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            const size_t subgraphs = multigraph->get_internal_subgraphs_size();
            for (size_t idx = 0; idx < subgraphs; ++idx) {
                collect_from_model(multigraph->get_function(idx), extractors, report);
            }
        }
    }
}

}  // namespace

FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs) {
    return build_framework_node_error_info(attrs, std::string{}, std::string{}, std::string{});
}

FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs,
                                                       const std::string& op_type_attr_key,
                                                       const std::string& failure_attr_key) {
    return build_framework_node_error_info(attrs, op_type_attr_key, failure_attr_key, std::string{});
}

FrameworkNodeErrorInfo build_framework_node_error_info(const ov::op::util::FrameworkNodeAttrs& attrs,
                                                       const std::string& op_type_attr_key,
                                                       const std::string& failure_attr_key,
                                                       const std::string& version_attr_key) {
    FrameworkNodeErrorInfo info;
    auto find_attr = [&](const std::string& key) -> std::string {
        const auto it = attrs.find(key);
        return it == attrs.end() ? std::string{} : it->second;
    };

    info.op_type = op_type_attr_key.empty() ? std::string{} : find_attr(op_type_attr_key);
    if (info.op_type.empty()) {
        info.op_type = attrs.get_type_name();
    }
    if (info.op_type.empty()) {
        info.op_type = "<unknown>";
    }

    const auto version = version_attr_key.empty() ? std::string{} : find_attr(version_attr_key);
    if (!version.empty()) {
        info.op_type += "-" + version;
    }

    const auto& opset_name = attrs.get_opset_name();
    if (!opset_name.empty()) {
        info.op_type = opset_name + "." + info.op_type;
    }

    if (!failure_attr_key.empty()) {
        info.failure_message = find_attr(failure_attr_key);
    }

    return info;
}

UnconvertedOpMap collect_unconverted_ops(const std::shared_ptr<ov::Model>& model,
                                         const std::vector<UnconvertedOpExtractor>& custom_extractors) {
    UnconvertedOpMap report;
    collect_from_model(model, custom_extractors, report);
    return report;
}

std::string format_unconverted_ops_report(const UnconvertedOpMap& unconverted_ops,
                                          const std::string& additional_error,
                                          const std::string& header) {
    std::stringstream error_msg;
    error_msg << (header.empty() ? "Model wasn't fully converted." : header);

    std::stringstream unsupported_msg;
    std::stringstream failures_msg;
    std::stringstream failed_short;
    unsupported_msg << "-- No conversion rule found for operations: ";
    failures_msg << " Failed operations detailed log:";
    failed_short << "-- Conversion is failed for: ";

    bool has_unsupported = false;
    bool has_failures = false;
    for (const auto& op : unconverted_ops) {
        if (op.second.empty()) {
            if (has_unsupported) {
                unsupported_msg << ", ";
            }
            unsupported_msg << op.first;
            has_unsupported = true;
        } else {
            if (has_failures) {
                failed_short << ", ";
            }
            failed_short << op.first;
            failures_msg << "\n-- " << op.first << " with a message:\n" << op.second;
            has_failures = true;
        }
    }

    if (has_failures) {
        error_msg << failures_msg.str();
    }
    error_msg << "\nSummary:" << additional_error;
    if (has_unsupported) {
        error_msg << '\n' << unsupported_msg.str();
    }
    if (has_failures) {
        error_msg << '\n' << failed_short.str();
    }

    return error_msg.str();
}

}  // namespace frontend
}  // namespace ov
