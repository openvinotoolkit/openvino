// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/backend_capabilities.hpp"
#include "intel_npu/utils/zero/zero_capabilities.hpp"

namespace intel_npu {

/**
 * @brief Selects from the level zero capabilities what the plugin and the compiler adapters need.
 */
inline BackendCapabilities makeBackendCapabilities(const ZeroCapabilities& zeroCaps) {
    BackendCapabilities caps;
    caps.driverVersion = zeroCaps.driverVersion;
    caps.graphExtVersion = zeroCaps.graphExtVersion;
    caps.commandQueueExtVersion = zeroCaps.commandQueueExtVersion;
    caps.apiVersion = static_cast<uint32_t>(zeroCaps.zeApiVersion);
    caps.commandQueueExtSupported = zeroCaps.commandQueueExtSupported;
    caps.luidExtSupported = zeroCaps.luidExtSupported;
    caps.contextExtSupported = zeroCaps.contextExtSupported;
    return caps;
}

}  // namespace intel_npu
