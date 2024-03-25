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
#include "npu/utils/logger/logger.hpp"
#include "vpux.hpp"
#include "vpux_private_properties.hpp"

namespace vpux {

enum class AvailableBackends { LEVEL_ZERO, IMD };

/** @brief Represent container for all backends and hide all related searching logic */
class VPUXBackends final {
public:
    explicit VPUXBackends(const std::vector<AvailableBackends>& backendRegistry, const Config& config);

    std::shared_ptr<IDevice> getDevice(const std::string& specificName = "") const;
    std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const;
    std::vector<std::string> getAvailableDevicesNames() const;
    std::string getBackendName() const;
    void registerOptions(OptionsDesc& options) const;
    std::string getCompilationPlatform(const std::string_view platform, const std::string& deviceId) const;

    void setup(const Config& config);

private:
    intel_npu::Logger _logger;
    ov::SoPtr<IEngineBackend> _backend;
};

}  // namespace vpux
