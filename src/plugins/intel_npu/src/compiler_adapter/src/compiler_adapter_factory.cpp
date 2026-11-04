// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <exception>
#include <memory>
#include <optional>

#include "compiler_impl.hpp"
#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "plugin_compiler_adapter.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {

namespace {

// The device that will run the compiled model is also the compilation target, so the driver compiler can take over.
bool isOnlineCompilation(const std::shared_ptr<IDevice>& device, std::string_view platform) {
    return device && (platform.empty() || device->getName() == platform);
}

}  // namespace

CompilerAdapterFactory::CompilerAdapterFactory(const VCLFunctionTableProvider& vclFunctions)
    : _vclFunctions(vclFunctions) {
    OPENVINO_ASSERT(_vclFunctions != nullptr, "CompilerAdapterFactory requires a way to obtain the VCL entry points");
}

std::unique_ptr<ICompilerAdapter> CompilerAdapterFactory::getCompiler(
    const ov::SoPtr<IEngineBackend>& engineBackend,
    ov::intel_npu::CompilerType& compilerType,
    std::string_view platform,
    const std::shared_ptr<OptionSupportCache>& optionSupportCache) const {
    const auto device = engineBackend ? engineBackend->getDevice() : nullptr;
    const bool preferPlugin = compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN;

    if (preferPlugin) {
        compilerType = resolvePreferPluginCompiler(device, platform);
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        std::shared_ptr<const VCLFunctionTable> functions;
        std::shared_ptr<IVCLCompiler> compiler;
        try {
            functions = _vclFunctions();
            compiler = std::make_shared<VCLCompilerImpl>(
                functions,
                device ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()} : std::nullopt,
                ScopedOptionSupportCache{
                    optionSupportCache,
                    static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::PLUGIN)});
        } catch (const std::exception& vclException) {
            // Only a PREFER_PLUGIN request that the driver compiler can serve recovers. An explicit PLUGIN request
            // or an offline compilation has no alternative.
            if (!preferPlugin || !isOnlineCompilation(device, platform)) {
                OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
            }
            Logger::global().warning("Failed to create the plugin compiler, falling back to the driver compiler: %s",
                                     vclException.what());
            compilerType = ov::intel_npu::CompilerType::DRIVER;
        }

        if (compiler != nullptr) {
            std::shared_ptr<IProfilingDecoder> profilingDecoder;
            try {
                profilingDecoder = std::make_shared<VCLProfilingDecoder>(functions);
            } catch (const std::exception& ex) {
                Logger::global().warning("Failed to create the VCL profiling decoder for plugin compilation: %s",
                                         ex.what());
            }

            return std::make_unique<PluginCompilerAdapter>(engineBackend ? engineBackend->getInitStructs() : nullptr,
                                                           compiler,
                                                           profilingDecoder);
        }
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

    compilerType = resolvePreferPluginCompiler(device, platform);
}

ov::intel_npu::CompilerType CompilerAdapterFactory::determineAppropriateCompilerTypeBasedOnPlatform(
    std::string_view platform) const {
    if (platform == ov::intel_npu::Platform::NPU4000 || platform == ov::intel_npu::Platform::NPU5010 ||
        platform == ov::intel_npu::Platform::NPU5020 || platform == ov::intel_npu::Platform::NPU6010) {
        return ov::intel_npu::CompilerType::PLUGIN;
    }

    return ov::intel_npu::CompilerType::DRIVER;
}

ov::intel_npu::CompilerType CompilerAdapterFactory::resolvePreferPluginCompiler(
    const std::shared_ptr<intel_npu::IDevice>& device,
    std::string_view platform) const {
    const bool onlineCompilation = isOnlineCompilation(device, platform);

    if (onlineCompilation &&
        determineAppropriateCompilerTypeBasedOnPlatform(platform) == ov::intel_npu::CompilerType::DRIVER) {
        return ov::intel_npu::CompilerType::DRIVER;
    }

    // The actual compiler will be created in getCompiler(). The provider remembers a failed load, so asking
    // again does not repeat it.
    try {
        _vclFunctions();
    } catch (const std::exception& ex) {
        if (onlineCompilation) {
            return ov::intel_npu::CompilerType::DRIVER;
        }
        OPENVINO_THROW("Plugin compiler is unavailable for offline or cross compilation. Error: ", ex.what());
    }

    return ov::intel_npu::CompilerType::PLUGIN;
}

const std::vector<ov::intel_npu::CompilerType>& CompilerAdapterFactory::getKnownCompilerTypes() {
    static const std::vector<ov::intel_npu::CompilerType> knownCompiler = {ov::intel_npu::CompilerType::DRIVER,
                                                                           ov::intel_npu::CompilerType::PLUGIN};

    return knownCompiler;
}

}  // namespace intel_npu
