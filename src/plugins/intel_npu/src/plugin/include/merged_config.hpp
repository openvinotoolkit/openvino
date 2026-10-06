// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "intel_npu/config/config.hpp"
#include "openvino/core/any.hpp"

namespace intel_npu {

/**
 * @brief The configuration of a compile, query or import operation, as produced by the plugin property manager and
 * consumed by the compiled model.
 */
struct MergedConfig {
    // Runtime config, compile-time-only and internal compiler options are removed from it.
    Config runtimeConfig;
    // Compile-time, both-mode and internal compiler options supported by the resolved compiler, serialized as strings.
    // Includes the values set through set_property and environment variables, overridden by the merged properties.
    // Always empty on the import path, the model is already compiled.
    std::map<std::string, std::string> compilerProperties;
    // Properties unknown to the plugin, forwarded as they are to the compiled model.
    ov::AnyMap unknownProperties;
};

}  // namespace intel_npu
