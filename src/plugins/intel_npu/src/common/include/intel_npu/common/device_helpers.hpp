// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>

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
 */
std::shared_ptr<IDevice> getDeviceById(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& deviceId);

/**
 * @brief Gets the optimal number of infer requests in parallel for the given platform and performance mode.
 * @param platform The platform for which to get the optimal number of infer requests.
 * @param performanceMode The performance mode for which to get the optimal number of infer requests.
 * @return The optimal number of infer requests in parallel.
 * @details Heuristically obtained number. Varies depending on the values of PLATFORM and PERFORMANCE_HINT
 * Note: This is the value provided by the plugin, application should query and consider it, but may supply its own
 * preference for number of parallel requests via dedicated configuration
 */
uint32_t getOptimalNumberOfInferRequestsInParallel(std::string_view platform,
                                                   const ov::hint::PerformanceMode performanceMode);

}  // namespace utils

}  // namespace intel_npu
