// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/device_helpers.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

bool utils::isNPUDevice(const uint32_t deviceId) {
    // bits 26-24 define interface type
    // 000 - IPC
    // 001 - PCIe
    // 010 - USB
    // 011 - ethernet
    constexpr uint32_t INTERFACE_TYPE_SELECTOR = 0x7000000;
    uint32_t interfaceType = (deviceId & INTERFACE_TYPE_SELECTOR);
    return (interfaceType == 0);
}

uint32_t utils::getSliceIdBySwDeviceId(const uint32_t swDevId) {
    // bits 3-1 define slice ID
    // right shift to omit bit 0, thus slice id is stored in bits 2-0
    // apply b111 mask to discard anything but slice ID
    uint32_t sliceId = (swDevId >> 1) & 0x7;
    return sliceId;
}

std::string utils::getPlatformByDeviceName(const std::string_view deviceName) {
    const auto platformPos = deviceName.rfind('.');
    const std::string_view platformName =
        (platformPos == std::string::npos) ? deviceName : deviceName.substr(0, platformPos);

    return std::string(platformName);
}

std::string utils::getCompilationPlatform(const std::string_view platform,
                                          const std::string_view deviceId,
                                          std::vector<std::string> availableDevicesNames) {
    // Platform parameter has a higher priority than deviceID
    if (platform != ov::intel_npu::Platform::AUTO_DETECT) {
        return ov::intel_npu::Platform::standardize(platform);
    }

    // Get compilation platform from deviceID
    if (!deviceId.empty()) {
        return ov::intel_npu::Platform::standardize(utils::getPlatformByDeviceName(deviceId));
    }

    // Automatic detection of compilation platform
    if (availableDevicesNames.empty()) {
        OPENVINO_THROW("No NPU devices were found.");
    }

    return ov::intel_npu::Platform::standardize(utils::getPlatformByDeviceName(availableDevicesNames.at(0)));
}

ov::intel_npu::CompilerType utils::resolveCompilerType(const FilteredConfig& base_conf, const ov::AnyMap& local_conf) {
    // first look if provided config changes compiler type
    auto it = local_conf.find(ov::intel_npu::compiler_type.name());
    if (it != local_conf.end()) {
        // if compiler_type is provided by local config = use that
        return COMPILER_TYPE::parse(it->second.as<std::string>());
    }
    // if there is no compiler_type provided = use base_config value
    return base_conf.get<COMPILER_TYPE>();
}

std::string utils::resolvePlatformOption(const FilteredConfig& base_conf, const ov::AnyMap& local_conf) {
    auto platform = local_conf.find(ov::intel_npu::platform.name());
    if (platform != local_conf.end()) {
        return platform->second.as<std::string>();
    }
    return base_conf.get<PLATFORM>();
}

std::string utils::resolveDeviceIdOption(const FilteredConfig& base_conf, const ov::AnyMap& local_conf) {
    auto device_id = local_conf.find(std::string(ov::device::id.name()));
    if (device_id != local_conf.end()) {
        return device_id->second.as<std::string>();
    }
    return base_conf.get<DEVICE_ID>();
}

}  // namespace intel_npu
