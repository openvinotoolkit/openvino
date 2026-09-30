// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string_view>
#include <vector>

#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

namespace utils {
bool isNPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
std::string getPlatformByDeviceName(const std::string_view deviceName);

/**
 * @brief A platform this plugin's compiler can compile for offline, and the PCI device IDs it ships
 * under.
 */
struct KnownPlatform {
    std::string_view platform;
    std::vector<uint32_t> deviceIds;
};

/**
 * @brief Every platform this plugin's compiler can compile for offline, with the PCI device IDs each
 * one ships under. Single source of truth for both directions: getPlatformByDeviceId() below maps a
 * live device to its platform, while the offline_compilation_targets property enumerates the same
 * table with no device present.
 */
const std::vector<KnownPlatform>& getKnownPlatforms();

/**
 * @brief The standardized platform a PCI device ID belongs to, or an empty view if the ID is unknown.
 */
std::string_view getPlatformByDeviceId(uint32_t deviceId);

/**
 * @brief Resolves ov::compilation_target (if present in properties) into ov::intel_npu::platform,
 * before platform/device resolution runs. ov::compilation_target itself is left in properties and
 * never reaches the compiler.
 * @throws ov::Exception if properties also carries an explicit ov::intel_npu::platform naming a
 * different platform. Whether the target's platform needs more than one blob is a separate,
 * compiler-driven check - see ICompilerAdapter::resolve_compilation_target_bundles().
 */
void resolveCompilationTarget(ov::AnyMap& properties);
std::string getCompilationPlatform(const ov::SoPtr<IEngineBackend>& engineBackend,
                                   const std::string_view platform,
                                   const std::string_view deviceId);

/**
 * @brief Gets the device by its ID.
 */
std::shared_ptr<IDevice> getDeviceById(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& deviceId);
std::string getFullDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
IDevice::Uuid getDeviceUuid(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
ov::device::LUID getDeviceLUID(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getSteppingNumber(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getMaxTiles(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint64_t getDeviceAllocMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint64_t getDeviceTotalMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
std::string getDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
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
