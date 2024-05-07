// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin
#include "metrics.hpp"

#include "device_helpers.hpp"
#include "npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

Metrics::Metrics(const std::shared_ptr<const NPUBackends>& backends) : _backends(backends) {
    _supportedMetrics = {ov::supported_properties.name(),
                         ov::available_devices.name(),
                         ov::device::full_name.name(),
                         ov::device::capabilities.name(),
                         ov::range_for_async_infer_requests.name(),
                         ov::range_for_streams.name(),
                         ov::device::capability::EXPORT_IMPORT,
                         ov::device::architecture.name(),
                         ov::internal::caching_properties.name(),
                         ov::internal::supported_properties.name(),
                         ov::cache_dir.name(),
                         ov::intel_npu::device_alloc_mem_size.name(),
                         ov::intel_npu::device_total_mem_size.name(),
                         ov::intel_npu::driver_version.name()};

    _supportedConfigKeys = {ov::log::level.name(),
                            ov::enable_profiling.name(),
                            ov::device::id.name(),
                            ov::hint::performance_mode.name(),
                            ov::num_streams.name(),
                            ov::hint::num_requests.name(),
                            ov::intel_npu::compilation_mode_params.name(),
                            ov::intel_npu::dynamic_shape_to_static.name()};
}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backends == nullptr ? std::vector<std::string>() : _backends->getAvailableDevicesNames();
}

// TODO each backend may support different metrics
const std::vector<std::string>& Metrics::SupportedMetrics() const {
    return _supportedMetrics;
}

std::string Metrics::GetFullDeviceName(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getFullDeviceName();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

// TODO each backend may support different configs
const std::vector<std::string>& Metrics::GetSupportedConfigKeys() const {
    return _supportedConfigKeys;
}

// TODO each backend may support different optimization capabilities
const std::vector<std::string>& Metrics::GetOptimizationCapabilities() const {
    return _optimizationCapabilities;
}

const std::tuple<uint32_t, uint32_t, uint32_t>& Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& Metrics::GetRangeForStreams() const {
    return _rangeForStreams;
}

std::string Metrics::GetDeviceArchitecture(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    return utils::getPlatformByDeviceName(devName);
}

IDevice::Uuid Metrics::GetDeviceUuid(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getUuid();
    }
    return IDevice::Uuid{};
}

std::vector<ov::PropertyName> Metrics::GetCachingProperties() const {
    return _cachingProperties;
}

std::vector<ov::PropertyName> Metrics::GetInternalSupportedProperties() const {
    return _internalSupportedProperties;
}

std::string Metrics::GetBackendName() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getBackendName();
}

uint32_t Metrics::GetDriverVersion() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getDriverVersion();
}

uint32_t Metrics::GetDriverExtVersion() const {
    if (_backends == nullptr) {
        OPENVINO_THROW("No available backends");
    }

    return _backends->getDriverExtVersion();
}

uint64_t Metrics::GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getAllocMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

uint64_t Metrics::GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    auto device = _backends->getDevice(devName);
    if (device) {
        return device->getTotalMemSize();
    }
    OPENVINO_THROW("No device with name '", specifiedDeviceName, "' is available");
}

std::string Metrics::getDeviceName(const std::string& specifiedDeviceName) const {
    std::vector<std::string> devNames;
    if (_backends == nullptr || (devNames = _backends->getAvailableDevicesNames()).empty()) {
        OPENVINO_THROW("No available devices");
    }

    // In case of single device and empty input from user we should use the first element from the device list
    if (specifiedDeviceName.empty()) {
        if (devNames.size() == 1) {
            return devNames[0];
        } else {
            OPENVINO_THROW("The device name was not specified. Please specify device name by providing DEVICE_ID");
        }
    }

    return specifiedDeviceName;
}

}  // namespace intel_npu
