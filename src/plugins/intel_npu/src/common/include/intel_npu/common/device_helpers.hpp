// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <tuple>

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

std::vector<std::string> getAvailableDevicesNames(const ov::SoPtr<IEngineBackend>& engineBackend);
std::string getDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);

std::string getFullDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
IDevice::Uuid getDeviceUuid(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
ov::device::LUID getDeviceLUID(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
bool isLUIDSupported(const ov::SoPtr<IEngineBackend>& engineBackend);
std::string getDeviceArchitecture(const ov::SoPtr<IEngineBackend>& engineBackend,
                                  const std::string& specifiedDeviceName);
std::string getBackendName(const ov::SoPtr<IEngineBackend>& engineBackend);
uint64_t getDeviceAllocMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint64_t getDeviceTotalMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getDriverVersion(const ov::SoPtr<IEngineBackend>& engineBackend);
uint32_t getGraphExtVersion(const ov::SoPtr<IEngineBackend>& engineBackend);
uint32_t getSteppingNumber(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getMaxTiles(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
ov::device::PCIInfo getPciInfo(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
std::map<ov::element::Type, float> getGops(const ov::SoPtr<IEngineBackend>& engineBackend,
                                           const std::string& specifiedDeviceName);
ov::device::Type getDeviceType(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);

/**
 * @brief Gets the optimal number of infer requests in parallel for the given platform and performance mode.
 * @param platform The platform for which to get the optimal number of infer requests.
 * @param performanceMode The performance mode for which to get the optimal number of infer requests.
 * @return The optimal number of infer requests in parallel.
 * @note This is the value provided by the plugin, application should query and consider it, but may supply its own
 * preference for number of parallel requests via dedicated configuration
 */
uint32_t getOptimalNumberOfInferRequestsInParallel(std::string_view platform,
                                                   const ov::hint::PerformanceMode performanceMode);

}  // namespace utils

}  // namespace intel_npu
