// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace intel_npu {

/**
 * @brief What the plugin and the compiler adapters need to know about the active backend.
 * @details A plain value, independent of any backend implementation or driver API, so the plugin does not need the
 * backend-specific structures to take decisions. A backend that cannot provide a value leaves it at the default (0 or
 * false), which reads as "not available".
 */
struct BackendCapabilities {
    uint32_t driverVersion = 0;
    uint32_t graphExtVersion = 0;
    uint32_t commandQueueExtVersion = 0;
    // Level zero API version as reported by the driver
    uint32_t apiVersion = 0;

    bool commandQueueExtSupported = false;
    bool luidExtSupported = false;
    bool contextExtSupported = false;
};

}  // namespace intel_npu
