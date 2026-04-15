// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace frontend {

/// \brief Structure containing information about unconverted operations
/// Map of operation types with no conversion rule (op_type -> empty string)
/// or operations that failed during conversion (op_type -> error message)
struct UnconvertedOpsReport {
    std::map<std::string, std::string> unconverted_ops;

    bool has_issues() const;

    /// \brief Add an unconverted operation if not already present
    /// \param op_type Operation type name
    /// \param error_message Error message (empty if no converter found, non-empty if conversion failed)
    void add(const std::string& op_type, const std::string& error_message = {});

    /// \brief Merge another report into this one
    void merge(const UnconvertedOpsReport& other);
};

/// \brief Callback type for extracting unconverted operation info from framework-specific nodes
/// \param node The node to check
/// \return Optional pair of (op_type, error_message) if this is an unconverted framework node
using FrameworkNodeExtractor =
    std::function<std::optional<std::pair<std::string, std::string>>(const std::shared_ptr<ov::Node>&)>;

/// \brief Collect unconverted operations from a model
/// \param model The model to scan
/// \param extractor Callback to extract info from framework-specific nodes
/// \return Report containing all unconverted operations found
UnconvertedOpsReport collect_unconverted_ops(const std::shared_ptr<ov::Model>& model,
                                             const FrameworkNodeExtractor& extractor);

/// \brief Callback type for adding additional error information
/// \param unsupported_ops Set of unsupported operation types (those without error messages)
/// \return Additional message to append to the error report
using AdditionalErrorCallback = std::function<std::string(const std::set<std::string>&)>;

/// \brief Check conversion result, send telemetry, and optionally throw if there are unconverted operations
/// \param report The unconverted operations report
/// \param telemetry Telemetry extension (can be nullptr)
/// \param telemetry_prefix Frontend name prefix for telemetry events (e.g., "pytorch", "tf", "onnx", "jax")
/// \param error_message_prefix Prefix for filtering error messages in telemetry (e.g., "[PyTorch Frontend] ")
///                             If non-empty, error_info telemetry events will be sent
/// \param additional_error Additional error message (e.g., from normalize step)
/// \param additional_callback Optional callback for frontend-specific additional error messages
/// \param throw_on_issues If true (default), throws when there are unconverted ops or additional_error is non-empty
/// \throws ov::frontend::OpConversionFailure if throw_on_issues is true and there are issues
void check_unconverted_ops(const UnconvertedOpsReport& report,
                           const std::shared_ptr<TelemetryExtension>& telemetry,
                           const std::string& telemetry_prefix,
                           const std::string& error_message_prefix = {},
                           const std::string& additional_error = {},
                           const AdditionalErrorCallback& additional_callback = nullptr,
                           bool throw_on_issues = true);

}  // namespace frontend
}  // namespace ov
