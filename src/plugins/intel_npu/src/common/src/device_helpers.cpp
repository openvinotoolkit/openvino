// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/device_helpers.hpp"

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
        return std::string();
    }

    return ov::intel_npu::Platform::standardize(utils::getPlatformByDeviceName(availableDevicesNames.at(0)));
}

std::shared_ptr<IDevice> utils::getDeviceById(const ov::SoPtr<IEngineBackend>& engineBackend,
                                              const std::string& deviceId) {
    if (engineBackend == nullptr) {
        return nullptr;
    }

    try {
        return engineBackend->getDevice(deviceId);
    } catch (...) {
        Logger("getDeviceById", Logger::global().level())
            .warning("The specified device (\"%s\") was not found.", deviceId.c_str());
    }
    return nullptr;
}

std::vector<std::string> utils::getAvailableDevicesNames(const ov::SoPtr<IEngineBackend>& engineBackend) {
    return engineBackend == nullptr ? std::vector<std::string>() : engineBackend->getDeviceNames();
}

std::string utils::getDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend,
                                 const std::string& specifiedDeviceName) {
    // In case of single device and empty input from user we should use the first element from the device list.
    if (specifiedDeviceName.empty()) {
        const auto devNames = getAvailableDevicesNames(engineBackend);
        if (devNames.empty()) {
            OPENVINO_THROW("No available devices");
        }

        return devNames[0];
    }

    return specifiedDeviceName;
}

std::string utils::getFullDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend,
                                     const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getFullDeviceName();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

IDevice::Uuid utils::getDeviceUuid(const ov::SoPtr<IEngineBackend>& engineBackend,
                                   const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    const auto& deviceToUse = getDeviceById(engineBackend, devName);
    if (deviceToUse) {
        return deviceToUse->getUuid();
    }

    return IDevice::Uuid{};
}

ov::device::LUID utils::getDeviceLUID(const ov::SoPtr<IEngineBackend>& engineBackend,
                                      const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getLUID();
    }

    return ov::device::LUID{{
        0,
    }};
}

bool utils::isLUIDSupported(const ov::SoPtr<IEngineBackend>& engineBackend) {
    return engineBackend != nullptr && engineBackend->isLUIDExtSupported();
}

std::string utils::getDeviceArchitecture(const ov::SoPtr<IEngineBackend>& engineBackend,
                                         const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    return getPlatformByDeviceName(devName);
}

std::string utils::getBackendName(const ov::SoPtr<IEngineBackend>& engineBackend) {
    if (engineBackend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return engineBackend->getName();
}

uint64_t utils::getDeviceAllocMemSize(const ov::SoPtr<IEngineBackend>& engineBackend,
                                      const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getAllocMemSize();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t utils::getDeviceTotalMemSize(const ov::SoPtr<IEngineBackend>& engineBackend,
                                      const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getTotalMemSize();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint32_t utils::getDriverVersion(const ov::SoPtr<IEngineBackend>& engineBackend) {
    if (engineBackend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return engineBackend->getDriverVersion();
}

uint32_t utils::getGraphExtVersion(const ov::SoPtr<IEngineBackend>& engineBackend) {
    if (engineBackend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return engineBackend->getGraphExtVersion();
}

uint32_t utils::getSteppingNumber(const ov::SoPtr<IEngineBackend>& engineBackend,
                                  const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getSubDevId();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint32_t utils::getMaxTiles(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device) {
        return device->getMaxNumSlices();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

ov::device::PCIInfo utils::getPciInfo(const ov::SoPtr<IEngineBackend>& engineBackend,
                                      const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device != nullptr) {
        return device->getPciInfo();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::map<ov::element::Type, float> utils::getGops(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device != nullptr) {
        return device->getGops();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

ov::device::Type utils::getDeviceType(const ov::SoPtr<IEngineBackend>& engineBackend,
                                      const std::string& specifiedDeviceName) {
    const auto devName = getDeviceName(engineBackend, specifiedDeviceName);
    auto device = getDeviceById(engineBackend, devName);
    if (device != nullptr) {
        return device->getDeviceType();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint32_t utils::getOptimalNumberOfInferRequestsInParallel(std::string_view platform,
                                                          const ov::hint::PerformanceMode performanceMode) {
    if (performanceMode != ov::hint::PerformanceMode::THROUGHPUT) {
        return 1;
    }

    return (platform == ov::intel_npu::Platform::NPU3720) ? 4 : 8;
}

}  // namespace intel_npu
