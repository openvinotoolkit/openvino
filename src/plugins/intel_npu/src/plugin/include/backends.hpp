// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// System
#include <memory>
#include <set>
#include <vector>

#include "openvino/runtime/so_ptr.hpp"

// Plugin
#include "intel_npu/common/npu.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

enum class AvailableBackends { LEVEL_ZERO, IMD };

/** @brief Represent container for all backends and hide all related searching logic */
class NPUBackends final {
public:
    explicit NPUBackends(const std::vector<AvailableBackends>& backendRegistry, const Config& config);

    std::shared_ptr<IDevice> getDevice(const std::string& specificName = "") const;
    std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const;
    std::vector<std::string> getAvailableDevicesNames() const;
    ov::SoPtr<IEngineBackend> getIEngineBackend();
    std::string getBackendName() const;
    uint32_t getDriverVersion() const;
    uint32_t getGraphExtVersion() const;
    bool isBatchingSupported() const;
    bool isCommandQueueExtSupported() const;
    bool isLUIDExtSupported() const;
    void registerOptions(OptionsDesc& options) const;
    void* getContext() const;
    std::string getCompilationPlatform(const std::string_view platform, const std::string& deviceId) const;

    void setup(const Config& config);

private:
    Logger _logger;
    ov::SoPtr<IEngineBackend> _backend;
};

}  // namespace intel_npu
