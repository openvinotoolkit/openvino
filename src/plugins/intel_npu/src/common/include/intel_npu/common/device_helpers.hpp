// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

namespace utils {
bool isNPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
std::string getPlatformByDeviceName(const std::string_view deviceName);
std::string getCompilationPlatform(const std::string_view platform,
                                   const std::string_view deviceId,
                                   std::vector<std::string> availableDevicesNames);

/**
 * @brief Gets the device by its ID.
 * @details Accounts for various scenarios when user passes DEVICE_ID:
 * - For offline compilation (backend == nullptr), no further action is needed; only model compilation is allowed.
 * - For on-device compilation where DEVICE_ID != current device ID, compilation is allowed only with CiP.
 */
std::shared_ptr<IDevice> getDeviceById(const ov::SoPtr<IEngineBackend>& engineBackend,
                                       const std::string& deviceId,
                                       const ov::intel_npu::CompilerType compilerType);
}  // namespace utils

}  // namespace intel_npu
