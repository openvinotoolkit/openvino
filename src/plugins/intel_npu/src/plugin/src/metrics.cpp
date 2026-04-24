// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "metrics.hpp"

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

Metrics::Metrics(const ov::SoPtr<IEngineBackend>& backend) : _backend(backend) {}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backend == nullptr ? std::vector<std::string>() : _backend->getDeviceNames();
}

std::string Metrics::GetFullDeviceName(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backend->getDevice(devName);
    if (device) {
        return device->getFullDeviceName();
    }

    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

const std::vector<std::string> Metrics::GetOptimizationCapabilities() const {
    return _optimizationCapabilities;
}

const std::tuple<uint32_t, uint32_t, uint32_t>& Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

std::string Metrics::GetDeviceArchitecture(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    return utils::getPlatformByDeviceName(devName);
}

IDevice::Uuid Metrics::GetDeviceUuid(const std::string& specifiedDeviceName) const {
    const auto& devName = getDeviceName(specifiedDeviceName);
    const auto& deviceToUse = getDevice(devName);
    if (deviceToUse) {
        return deviceToUse->getUuid();
    }
    return IDevice::Uuid{};
}

ov::device::LUID Metrics::GetDeviceLUID(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device) {
        return device->getLUID();
    }
    return ov::device::LUID{{
        0,
    }};
}

std::string Metrics::GetBackendName() const {
    if (_backend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return _backend->getName();
}

uint32_t Metrics::GetDriverVersion() const {
    if (_backend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return _backend->getDriverVersion();
}

uint32_t Metrics::GetGraphExtVersion() const {
    if (_backend == nullptr) {
        OPENVINO_THROW("No available backend");
    }

    return _backend->getGraphExtVersion();
}

uint32_t Metrics::GetSteppingNumber(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device) {
        return device->getSubDevId();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint32_t Metrics::GetMaxTiles(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device) {
        return device->getMaxNumSlices();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t Metrics::GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device) {
        return device->getAllocMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t Metrics::GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device) {
        return device->getTotalMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::string Metrics::getDeviceName(const std::string& specifiedDeviceName) const {
    // In case of single device and empty input from user we should use the first element from the device list
    if (specifiedDeviceName.empty()) {
        std::vector<std::string> devNames;
        if (_backend == nullptr || (devNames = _backend->getDeviceNames()).empty()) {
            OPENVINO_THROW("No available devices");
        }
        if (devNames.size() >= 1) {
            return devNames[0];
        } else {
            OPENVINO_THROW("The device name was not specified. Please specify device name by providing DEVICE_ID");
        }
    }

    return specifiedDeviceName;
}

std::shared_ptr<intel_npu::IDevice> Metrics::getDevice(const std::string& specifiedDeviceName) const {
    std::shared_ptr<IDevice> deviceToUse;

    if (_backend != nullptr) {
        if (specifiedDeviceName.empty()) {
            return _backend->getDevice();
        } else {
            return _backend->getDevice(specifiedDeviceName);
        }
    }

    return nullptr;
}

ov::device::PCIInfo Metrics::GetPciInfo(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device != nullptr) {
        return device->getPciInfo();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::map<ov::element::Type, float> Metrics::GetGops(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device != nullptr) {
        return device->getGops();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

ov::device::Type Metrics::GetDeviceType(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = getDevice(devName);
    if (device != nullptr) {
        return device->getDeviceType();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

bool Metrics::IsCommandQueueExtSupported() const {
    if (_backend == nullptr) {
        OPENVINO_THROW("No available backend");
    }
    return _backend->isCommandQueueExtSupported();
}

}  // namespace intel_npu
