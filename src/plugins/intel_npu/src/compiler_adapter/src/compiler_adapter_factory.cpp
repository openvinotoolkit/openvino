// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string_view>

#include "compiler_impl.hpp"
#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "plugin_compiler_adapter.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {
namespace {

std::shared_ptr<IVCLCompiler> makePluginCompiler(const std::shared_ptr<const VCLFunctionTable>& functions,
                                                 const std::shared_ptr<IDevice>& device,
                                                 std::string_view platform,
                                                 const std::shared_ptr<OptionSupportCache>& optionSupportCache) {
    const bool onlineCompilation = device && (platform.empty() || device->getName() == platform);
    return std::make_shared<VCLCompilerImpl>(
        functions,
        onlineCompilation ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()} : std::nullopt,
        ScopedOptionSupportCache{optionSupportCache,
                                 static_cast<OptionSupportCache::CacheKey>(ov::intel_npu::CompilerType::PLUGIN)});
}

}  // namespace

CompilerAdapterFactory::CompilerAdapterFactory(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                                                             const VCLFunctionTableProvider& vclFunctionsProvider)
        : _engineBackend(engineBackend),
            _vclFunctionsProvider(vclFunctionsProvider) {}

std::unique_ptr<ICompilerAdapter> CompilerAdapterFactory::getCompiler(
    ov::intel_npu::CompilerType& compilerType,
    std::string_view platform,
    const std::shared_ptr<OptionSupportCache>& optionSupportCache) const {
    const auto device = _engineBackend ? _engineBackend->getDevice() : nullptr;

    if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        compilerType = resolvePreferPluginCompiler(device, platform);
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        std::shared_ptr<const VCLFunctionTable> functions;
        std::shared_ptr<IVCLCompiler> compiler;
        try {
            OPENVINO_ASSERT(_vclFunctionsProvider != nullptr,
                            "CompilerAdapterFactory requires a way to obtain the VCL entry points");
            functions = _vclFunctionsProvider();
            compiler = makePluginCompiler(functions, device, platform, optionSupportCache);
        } catch (const std::exception& vclException) {
            OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
        }

        std::shared_ptr<IProfilingDecoder> profilingDecoder;
        try {
            profilingDecoder = std::make_shared<VCLProfilingDecoder>(functions);
        } catch (const std::exception& ex) {
            Logger::global().warning("Failed to create the VCL profiling decoder for plugin compilation: %s",
                                     ex.what());
        }

        return std::make_unique<PluginCompilerAdapter>(_engineBackend ? _engineBackend->getInitStructs() : nullptr,
                                                       compiler,
                                                       profilingDecoder);
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

        return std::make_unique<DriverCompilerAdapter>(_engineBackend->getInitStructs(), optionSupportCache);
    }

    OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
}

void CompilerAdapterFactory::decideCompilerType(ov::intel_npu::CompilerType& compilerType,
                                                const std::shared_ptr<intel_npu::IDevice>& device,
                                                std::string_view platform) const {
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
    const bool onlineCompilation = device && (platform.empty() || device->getName() == platform);

    if (onlineCompilation &&
        determineAppropriateCompilerTypeBasedOnPlatform(platform) == ov::intel_npu::CompilerType::DRIVER) {
        return ov::intel_npu::CompilerType::DRIVER;
    }

    try {
        OPENVINO_ASSERT(_vclFunctionsProvider != nullptr,
                        "CompilerAdapterFactory requires a way to obtain the VCL entry points");
        const auto functions = _vclFunctionsProvider();
        OPENVINO_ASSERT(functions != nullptr, "VCL function table provider returned no entry points");
    } catch (...) {
        if (onlineCompilation) {
            return ov::intel_npu::CompilerType::DRIVER;
        }
        OPENVINO_THROW("Failed to load VCL function table for offline or cross compilation.");
    }

    return ov::intel_npu::CompilerType::PLUGIN;
}

const std::vector<ov::intel_npu::CompilerType>& CompilerAdapterFactory::getKnownCompilerTypes() {
    static const std::vector<ov::intel_npu::CompilerType> knownCompiler = {ov::intel_npu::CompilerType::DRIVER,
                                                                           ov::intel_npu::CompilerType::PLUGIN};

    return knownCompiler;
}

}  // namespace intel_npu
