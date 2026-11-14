// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include <ze_command_queue_npu_ext.h>
#include <ze_context_npu_ext.h>

#include <cstdint>
#include <map>
#include <string>

namespace intel_npu {

/**
 * @brief Plain description of what the driver supports, decoupled from the driver itself.
 * @details Built once after the driver has been queried. It is a value type: copyable and cheap, so components can take
 * only what they need and tests can fill it in directly, without a driver.
 */
struct ZeroCapabilities {
    ze_api_version_t zeApiVersion = {};
    // Negotiated version of the mutable command list extension, 0 if not supported
    uint32_t mutableCommandListExtVersion = 0;
    uint32_t commandQueueGroupOrdinal = 0;

    bool commandQueueExtSupported = false;
    bool commandQueueSetPrioritySupported = false;
    bool contextExtSupported = false;
    bool luidExtSupported = false;
    bool externalMemoryStandardAllocationSupported = false;
    bool externalMemoryFdWin32Supported = false;
};

/**
 * @brief Builds the capabilities from the extensions negotiated between plugin and driver.
 * @param supportedExtensions Extension name (without version suffix) to negotiated version, as kept by the holder.
 * Extensions that are not present are treated as not supported.
 * @param externalMemoryStandardAllocationSupported Result of the external memory discovery (device properties and the
 * external memory mapping extension), it is not derived from the map.
 * @param externalMemoryFdWin32Supported Result of the external memory discovery (device properties).
 * @param commandQueueGroupOrdinal Ordinal of the compute command queue group.
 * @param zeApiVersion Level zero API version reported by the driver.
 */
inline ZeroCapabilities makeZeroCapabilities(const std::map<std::string, uint32_t, std::less<>>& supportedExtensions,
                                             bool externalMemoryStandardAllocationSupported,
                                             bool externalMemoryFdWin32Supported,
                                             uint32_t commandQueueGroupOrdinal,
                                             ze_api_version_t zeApiVersion) {
    const auto extensionVersion = [&](const char* name) -> uint32_t {
        const auto iter = supportedExtensions.find(name);
        return iter == supportedExtensions.end() ? 0 : iter->second;
    };

    ZeroCapabilities caps;
    caps.zeApiVersion = zeApiVersion;
    caps.mutableCommandListExtVersion = extensionVersion(ZE_MUTABLE_COMMAND_LIST_EXP_NAME);
    caps.commandQueueGroupOrdinal = commandQueueGroupOrdinal;
    caps.commandQueueExtSupported = extensionVersion(ZE_COMMAND_QUEUE_NPU_EXT_NAME) != 0;
    caps.commandQueueSetPrioritySupported = extensionVersion(ZE_COMMAND_QUEUE_SET_PRIORITY_EXT_NAME) != 0;
    caps.contextExtSupported = extensionVersion(ZE_CONTEXT_NPU_EXT_NAME) != 0;
    caps.luidExtSupported = extensionVersion(ZE_DEVICE_LUID_EXT_NAME) != 0;
    caps.externalMemoryStandardAllocationSupported = externalMemoryStandardAllocationSupported;
    caps.externalMemoryFdWin32Supported = externalMemoryFdWin32Supported;
    return caps;
}

}  // namespace intel_npu
