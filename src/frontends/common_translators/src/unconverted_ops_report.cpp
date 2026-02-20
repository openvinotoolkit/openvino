// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unconverted_ops_report.hpp"

#include <sstream>

#include "openvino/frontend/exception.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace frontend {

bool UnconvertedOpsReport::has_issues() const {
    return !unconverted_ops.empty();
}

void UnconvertedOpsReport::add(const std::string& op_type, const std::string& error_message) {
    if (unconverted_ops.find(op_type) == unconverted_ops.end()) {
        unconverted_ops[op_type] = error_message;
    }
}

void UnconvertedOpsReport::merge(const UnconvertedOpsReport& other) {
    for (const auto& [op_type, msg] : other.unconverted_ops) {
        add(op_type, msg);
    }
}

UnconvertedOpsReport collect_unconverted_ops(const std::shared_ptr<ov::Model>& model,
                                             const FrameworkNodeExtractor& extractor) {
    UnconvertedOpsReport report;
    if (!model) {
        return report;
    }

    for (const auto& node : model->get_ordered_ops()) {
        // Try framework-specific extractor first
        if (auto result = extractor(node)) {
            report.add(result->first, result->second);
        }

        // Handle MultiSubGraphOp (parent of Loop, If, etc.) - common for all frontends
        if (const auto& subgraph_op = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            for (size_t i = 0; i < subgraph_op->get_internal_subgraphs_size(); ++i) {
                report.merge(collect_unconverted_ops(subgraph_op->get_function(i), extractor));
            }
        }
    }
    return report;
}

namespace {

std::string format_unconverted_ops_report(const UnconvertedOpsReport& report,
                                          const std::string& additional_error,
                                          const AdditionalErrorCallback& additional_callback) {
    std::stringstream error_msg;
    std::stringstream unconverted_ops_msg;
    std::stringstream failed_ops_msg;
    std::stringstream failed_ops_short;

    error_msg << "Model wasn't fully converted.";
    unconverted_ops_msg << "-- No conversion rule found for operations: ";
    failed_ops_msg << " Failed operations detailed log:";
    failed_ops_short << "-- Conversion is failed for: ";

    bool has_unsupported = false;
    bool has_failed = false;
    std::set<std::string> unsupported_ops_set;

    for (const auto& [op_type, error] : report.unconverted_ops) {
        if (error.empty()) {
            // No conversion rule found
            if (has_unsupported) {
                unconverted_ops_msg << ", ";
            }
            unconverted_ops_msg << op_type;
            unsupported_ops_set.insert(op_type);
            has_unsupported = true;
        } else {
            // Conversion failed with error
            if (has_failed) {
                failed_ops_short << ", ";
            }
            failed_ops_short << op_type;
            failed_ops_msg << "\n-- " << op_type << " with a message:\n" << error;
            has_failed = true;
        }
    }

    if (has_failed) {
        error_msg << failed_ops_msg.str();
    }

    error_msg << "\nSummary:" << additional_error;

    if (has_unsupported) {
        error_msg << '\n' << unconverted_ops_msg.str();
    }

    if (has_failed) {
        error_msg << '\n' << failed_ops_short.str();
    }

    // Add additional callback-provided information
    if (additional_callback && has_unsupported) {
        if (auto additional_info = additional_callback(unsupported_ops_set); !additional_info.empty()) {
            error_msg << '\n' << additional_info;
        }
    }

    return error_msg.str();
}

}  // namespace

void check_unconverted_ops(const UnconvertedOpsReport& report,
                           const std::shared_ptr<TelemetryExtension>& telemetry,
                           const std::string& telemetry_prefix,
                           const std::string& error_message_prefix,
                           const std::string& additional_error,
                           const AdditionalErrorCallback& additional_callback,
                           bool throw_on_issues) {
    // Send telemetry for all unconverted operations
    if (telemetry) {
        const bool send_error_info = !error_message_prefix.empty();
        for (const auto& [op_type, error] : report.unconverted_ops) {
            telemetry->send_event("error_cause", telemetry_prefix + "_" + op_type);
            if (send_error_info && !error.empty()) {
                auto cropped_message = ov::util::filter_lines_by_prefix(error, error_message_prefix);
                if (!cropped_message.empty()) {
                    telemetry->send_event("error_info", cropped_message);
                }
            }
        }
    }

    if (throw_on_issues && (report.has_issues() || !additional_error.empty())) {
        FRONT_END_OP_CONVERSION_CHECK(false,
                                      format_unconverted_ops_report(report, additional_error, additional_callback));
    }
}

}  // namespace frontend
}  // namespace ov
