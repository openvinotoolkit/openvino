// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <string>

#include "compiler_impl.hpp"
#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/compiler_selection.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

namespace {
// Loads the compiler-in-plugin, translating any failure into the aborting message callers expect.
// Composing the compiler here, rather than inside PluginCompilerAdapter, keeps the adapter free of
// any knowledge of how a compiler is obtained.
std::shared_ptr<IVCLCompiler> makePluginCompiler(const std::shared_ptr<IDevice>& device,
                                                 const std::shared_ptr<OptionSupportCache>& optionSupportCache) {
    try {
        return makeVCLCompiler(
            device ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()} : std::nullopt,
            optionSupportCache);
    } catch (const std::exception& vclException) {
        OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
    }
}
}  // namespace

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
        return std::make_unique<PluginCompilerAdapter>(engineBackend ? engineBackend->getInitStructs() : nullptr,
                                                       makePluginCompiler(device, optionSupportCache));
    }

    if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        if (device == nullptr) {
            OPENVINO_THROW("Could not find an NPU device. The driver compiler requires a valid device to be present in "
                           "the system.");
        }

        // The driver compiles for the platform set by the user so a device from another one cannot be used.
        auto deviceName = device->getName();
        const auto normalizedPlatform = ov::intel_npu::Platform::standardize(platform);
        const auto normalizedDeviceName = ov::intel_npu::Platform::standardize(deviceName);
        if (!normalizedPlatform.empty() && normalizedDeviceName != normalizedPlatform &&
            normalizedDeviceName != ov::intel_npu::Platform::AUTO_DETECT) {
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

std::pair<std::unique_ptr<ICompilerAdapter>, ov::intel_npu::CompilerType>
CompilerAdapterFactory::resolvePreferPluginCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                    const std::shared_ptr<OptionSupportCache>& optionSupportCache,
                                                    const std::shared_ptr<intel_npu::IDevice>& device,
                                                    std::string_view platform) const {
    const auto normalizedDevicePlatform =
        device ? ov::intel_npu::Platform::standardize(device->getName()) : std::string{};
    const auto normalizedTargetPlatform =
        (platform.empty() || platform == ov::intel_npu::Platform::AUTO_DETECT)
            ? normalizedDevicePlatform
            : ov::intel_npu::Platform::standardize(platform);
    const bool onlineCompilation = device && !normalizedTargetPlatform.empty() &&
                                   normalizedTargetPlatform == normalizedDevicePlatform;

    if (onlineCompilation) {
#if defined(_WIN32) || defined(__CYGWIN__)
        constexpr bool windowsDriver = true;
#else
        constexpr bool windowsDriver = false;
#endif
        std::optional<uint32_t> driverVersion;
        if (normalizedTargetPlatform == ov::intel_npu::Platform::NPU3720) {
            driverVersion = device->getDeviceProperties().driverVersion;
        }

        if (!isPluginCompilerPreferred(normalizedTargetPlatform, driverVersion, windowsDriver)) {
            return {nullptr, ov::intel_npu::CompilerType::DRIVER};
        }
    }

    const auto pluginCompilerPresence = _pluginCompilerPresence.load(std::memory_order_acquire);

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
            auto pluginCompiler =
                std::make_unique<PluginCompilerAdapter>(engineBackend ? engineBackend->getInitStructs() : nullptr,
                                                        makePluginCompiler(device, optionSupportCache));
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
