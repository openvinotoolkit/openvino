// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "intel_npu/common/npu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class Metrics final {
public:
    Metrics(const ov::SoPtr<IEngineBackend>& backend);

    std::vector<std::string> GetAvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;
    std::string GetFullDeviceName(const std::string& specifiedDeviceName) const;
    IDevice::Uuid GetDeviceUuid(const std::string& specifiedDeviceName) const;
    ov::device::LUID GetDeviceLUID(const std::string& specifiedDeviceName) const;
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    const std::vector<std::string> GetOptimizationCapabilities() const;
    const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() const;
    const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() const;
    std::string GetDeviceArchitecture(const std::string& specifiedDeviceName) const;
    std::string GetBackendName() const;
    uint64_t GetDeviceAllocMemSize(const std::string& specifiedDeviceName) const;
    uint64_t GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const;
    uint32_t GetDriverVersion() const;
    uint32_t GetGraphExtVersion() const;
    uint32_t GetSteppingNumber(const std::string& specifiedDeviceName) const;
    uint32_t GetMaxTiles(const std::string& specifiedDeviceName) const;
    ov::device::PCIInfo GetPciInfo(const std::string& specifiedDeviceName) const;
    std::map<ov::element::Type, float> GetGops(const std::string& specifiedDeviceName) const;
    ov::device::Type GetDeviceType(const std::string& specifiedDeviceName) const;

    std::vector<ov::PropertyName> GetCachingProperties() const;
    std::vector<ov::PropertyName> GetInternalSupportedProperties() const;

    ~Metrics() = default;

private:
    const ov::SoPtr<IEngineBackend> _backend;
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    const std::vector<std::string> _optimizationCapabilities = {
        ov::device::capability::FP16,
        ov::device::capability::INT8,
        ov::device::capability::EXPORT_IMPORT,
    };
    const std::vector<ov::PropertyName> _cachingProperties = {ov::device::architecture.name(),
                                                              ov::intel_npu::compilation_mode_params.name(),
                                                              ov::intel_npu::compiler_dynamic_quantization.name(),
                                                              ov::intel_npu::tiles.name(),
                                                              ov::intel_npu::dpu_groups.name(),
                                                              ov::intel_npu::dma_engines.name(),
                                                              ov::intel_npu::compilation_mode.name(),
                                                              ov::intel_npu::driver_version.name(),
                                                              ov::intel_npu::compiler_type.name(),
                                                              ov::intel_npu::batch_mode.name(),
                                                              ov::hint::execution_mode.name()};

    const std::vector<ov::PropertyName> _internalSupportedProperties = {ov::internal::caching_properties.name(),
                                                                        ov::internal::caching_with_mmap.name()};

    // Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
    const std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests{1u, 10u, 1u};

    // Metric to provide information about a range for streams.(bottom bound, upper bound)
    const std::tuple<uint32_t, uint32_t> _rangeForStreams{1u, 4u};

    std::string getDeviceName(const std::string& specifiedDeviceName) const;
    std::shared_ptr<intel_npu::IDevice> getDevice(const std::string& specifiedDeviceName) const;
};

}  // namespace intel_npu
