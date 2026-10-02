// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

std::unique_ptr<ICompilerAdapter> CompilerAdapterFactory::getCompiler(
    const ov::SoPtr<IEngineBackend>& engineBackend,
    ov::intel_npu::CompilerType& compilerType,
    std::string_view platform,
    const std::shared_ptr<OptionSupportCache>& optionSupportCache) const {
    const auto device = engineBackend ? engineBackend->getDevice() : nullptr;

    if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        auto [pluginCompiler, resolvedCompilerType] =
            resolvePreferPluginCompiler(engineBackend, optionSupportCache, device, platform);
        compilerType = resolvedCompilerType;
        if (pluginCompiler) {
            return std::move(pluginCompiler);
        }
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        return std::make_unique<PluginCompilerAdapter>(
            engineBackend ? engineBackend->getInitStructs() : nullptr,
            optionSupportCache,
            device ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()} : std::nullopt);
    }

    if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
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
    }

    OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
}

void CompilerAdapterFactory::decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                                                const std::shared_ptr<intel_npu::IDevice>& device,
                                                std::string_view platform) {
    if (compilerType != ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        return;
    }

    compilerType = resolvePreferPluginCompiler({}, nullptr, device, platform).second;
}

ov::intel_npu::CompilerType CompilerAdapterFactory::determineAppropriateCompilerTypeBasedOnPlatform(
    std::string_view platform) const {
    if (platform == ov::intel_npu::Platform::NPU4000 || platform == ov::intel_npu::Platform::NPU5010 ||
        platform == ov::intel_npu::Platform::NPU5020 || platform == ov::intel_npu::Platform::NPU6010) {
        return ov::intel_npu::CompilerType::PLUGIN;
    }

    return ov::intel_npu::CompilerType::DRIVER;
}

std::pair<std::unique_ptr<ICompilerAdapter>, ov::intel_npu::CompilerType>
CompilerAdapterFactory::resolvePreferPluginCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                    const std::shared_ptr<OptionSupportCache>& optionSupportCache,
                                                    const std::shared_ptr<intel_npu::IDevice>& device,
                                                    std::string_view platform) const {
    const auto pluginCompilerPresence = _pluginCompilerPresence.load(std::memory_order_acquire);
    const bool onlineCompilation = device && (platform.empty() || device->getName() == platform);

    if (onlineCompilation &&
        determineAppropriateCompilerTypeBasedOnPlatform(platform) == ov::intel_npu::CompilerType::DRIVER) {
        return {nullptr, ov::intel_npu::CompilerType::DRIVER};
    }

    if (pluginCompilerPresence == PluginCompilerPresence::PRESENT) {
        // The actual compiler will be created in getCompiler().
        return {nullptr, ov::intel_npu::CompilerType::PLUGIN};
    }

    if (pluginCompilerPresence == PluginCompilerPresence::ABSENT) {
        if (onlineCompilation) {
            return {nullptr, ov::intel_npu::CompilerType::DRIVER};
        }
        OPENVINO_THROW("Plugin compiler is absent for offline or cross compilation.");
    }

    if (pluginCompilerPresence == PluginCompilerPresence::UNKNOWN) {
        try {
            auto pluginCompiler = std::make_unique<PluginCompilerAdapter>(
                engineBackend ? engineBackend->getInitStructs() : nullptr,
                optionSupportCache,
                device ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()} : std::nullopt);
            _pluginCompilerPresence.store(PluginCompilerPresence::PRESENT, std::memory_order_release);
            return {std::move(pluginCompiler), ov::intel_npu::CompilerType::PLUGIN};
        } catch (...) {
            _pluginCompilerPresence.store(PluginCompilerPresence::ABSENT, std::memory_order_release);
            if (onlineCompilation) {
                return {nullptr, ov::intel_npu::CompilerType::DRIVER};
            }
            OPENVINO_THROW("Failed to create plugin compiler for offline or cross compilation.");
        }
    }

    // Should not reach here, but throw in case of unexpected state.
    OPENVINO_THROW("Unexpected state in resolvePreferPluginCompiler");
}

const std::vector<ov::intel_npu::CompilerType>& CompilerAdapterFactory::getKnownCompilerTypes() {
    static const std::vector<ov::intel_npu::CompilerType> knownCompiler = {ov::intel_npu::CompilerType::DRIVER,
                                                                           ov::intel_npu::CompilerType::PLUGIN};

    return knownCompiler;
}

}  // namespace intel_npu
