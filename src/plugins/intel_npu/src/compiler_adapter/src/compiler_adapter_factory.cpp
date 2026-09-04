// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

ov::intel_npu::CompilerType CompilerAdapterFactory::determineAppropriateCompilerTypeBasedOnPlatform(
    std::string_view platform) const {
    if (platform == ov::intel_npu::Platform::NPU4000 || platform == ov::intel_npu::Platform::NPU5010 ||
        platform == ov::intel_npu::Platform::NPU5020 || platform == ov::intel_npu::Platform::NPU6010) {
        return ov::intel_npu::CompilerType::PLUGIN;
    }

    return ov::intel_npu::CompilerType::DRIVER;
}

std::unique_ptr<ICompilerAdapter> CompilerAdapterFactory::getCompiler(
    const ov::SoPtr<IEngineBackend>& engineBackend,
    ov::intel_npu::CompilerType& compilerType,
    std::string_view platform,
    const std::shared_ptr<OptionSupportCache>& optionSupportCache) const {
    const auto device = engineBackend != nullptr ? engineBackend->getDevice() : nullptr;

    if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        if (device != nullptr) {
            compilerType = determineAppropriateCompilerTypeBasedOnPlatform(platform);
            if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
                const auto pluginCompilerPresence = _pluginCompilerPresence.load(std::memory_order_acquire);
                if (pluginCompilerPresence == PluginCompilerPresence::ABSENT) {
                    // plugin compiler isn't present, fallback to driver compiler
                    compilerType = ov::intel_npu::CompilerType::DRIVER;
                } else {
                    try {
                        auto pluginCompiler = std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs(),
                                                                                      optionSupportCache,
                                                                                      device->getDeviceProperties());
                        _pluginCompilerPresence.store(PluginCompilerPresence::PRESENT, std::memory_order_release);
                        return pluginCompiler;
                    } catch (...) {
                        _pluginCompilerPresence.store(PluginCompilerPresence::ABSENT, std::memory_order_release);
                        compilerType = ov::intel_npu::CompilerType::DRIVER;
                    }
                }
            }
        } else {
            // device isn't available, offline compilation only
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
        }
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        if (device == nullptr) {
            return std::make_unique<PluginCompilerAdapter>(nullptr, optionSupportCache);
        }

        return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs(),
                                                       optionSupportCache,
                                                       device->getDeviceProperties());
    } else if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        if (device == nullptr) {
            OPENVINO_THROW("Could not find an NPU device. The driver compiler requires a valid device to be present in "
                           "the system.");
        }

        // It is required to check if the device is compatible with the provided platform, as the driver compiler
        // will be used.
        auto deviceName = device->getName();
        if (!platform.empty() && deviceName != platform && deviceName != "AUTO_DETECT") {
            OPENVINO_THROW("Could not find a valid NPU device for the provided configuration.");
        }

        return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs(), optionSupportCache);
    } else {
        OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
    }
}

void CompilerAdapterFactory::decideCompilerType(ov::intel_npu::CompilerType& compilerType, std::string_view platform) {
    if (compilerType != ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        return;
    }

    const auto pluginCompilerPresence = _pluginCompilerPresence.load(std::memory_order_acquire);
    if (pluginCompilerPresence == PluginCompilerPresence::ABSENT) {
        compilerType = ov::intel_npu::CompilerType::DRIVER;
        return;
    } else if (pluginCompilerPresence == PluginCompilerPresence::UNKNOWN) {
        compilerType = determineAppropriateCompilerTypeBasedOnPlatform(platform);
        if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
            return;
        }

        try {
            (void)std::make_unique<PluginCompilerAdapter>(nullptr);
            _pluginCompilerPresence.store(PluginCompilerPresence::PRESENT, std::memory_order_release);
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
            return;
        } catch (...) {
            _pluginCompilerPresence.store(PluginCompilerPresence::ABSENT, std::memory_order_release);
            compilerType = ov::intel_npu::CompilerType::DRIVER;
            return;
        }
    }

    compilerType = ov::intel_npu::CompilerType::PLUGIN;
}

const std::vector<ov::intel_npu::CompilerType>& CompilerAdapterFactory::getKnownCompilerTypes() {
    static const std::vector<ov::intel_npu::CompilerType> knownCompiler = {ov::intel_npu::CompilerType::DRIVER,
                                                                           ov::intel_npu::CompilerType::PLUGIN};

    return knownCompiler;
}

}  // namespace intel_npu
