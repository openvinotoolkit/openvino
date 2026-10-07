// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include "compiler_impl.hpp"
#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "plugin_compiler_adapter.hpp"
#include "vcl_profiling_decoder.hpp"

namespace intel_npu {

namespace {

// Resolves the entry points and composes the compiler-in-plugin over them, translating any failure
// into the aborting message callers expect. Composing here, rather than inside
// PluginCompilerAdapter, keeps the adapter free of any knowledge of how a compiler is obtained.
//
// This is also the only place the provider is invoked, so the library is loaded if and only if a
// compiler-in-plugin is actually being built.
std::shared_ptr<IVCLCompiler> makePluginCompiler(const VCLFunctionTableProvider& vclFunctions,
                                                 const std::shared_ptr<IDevice>& device,
                                                 const std::shared_ptr<OptionSupportCache>& optionSupportCache) {
    try {
        auto compiler = makeVCLCompiler(vclFunctions(),
                                        device ? std::optional<IDevice::DeviceProperties>{device->getDeviceProperties()}
                                               : std::nullopt,
                                        optionSupportCache);
        return compiler._ptr;
    } catch (const std::exception& vclException) {
        OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
    }
}

}  // namespace

CompilerAdapterFactory::CompilerAdapterFactory(VCLFunctionTableProvider vclFunctions)
    : _vclFunctions(std::move(vclFunctions)) {
    OPENVINO_ASSERT(_vclFunctions != nullptr, "CompilerAdapterFactory requires a way to obtain the VCL entry points");
}

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
        std::shared_ptr<IVCLCompiler> compiler;
        try {
            compiler = makePluginCompiler(_vclFunctions, device, optionSupportCache);
        } catch (const std::exception& vclException) {
            OPENVINO_THROW("VCL compiler loading failed, aborting. Error: ", vclException.what());
        }

        std::shared_ptr<IProfilingDecoder> profilingDecoder;
        try {
            profilingDecoder = makeVCLProfilingDecoder(_vclFunctions());
        } catch (const std::exception& ex) {
            Logger::global().warning("Failed to create the VCL profiling decoder for plugin compilation: %s",
                                     ex.what());
        }

        return std::make_unique<PluginCompilerAdapter>(engineBackend ? engineBackend->getInitStructs() : nullptr,
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
        std::shared_ptr<IVCLCompiler> pluginCompiler;
        try {
            pluginCompiler = makePluginCompiler(_vclFunctions, device, optionSupportCache);
        } catch (...) {
            _pluginCompilerPresence.store(PluginCompilerPresence::ABSENT, std::memory_order_release);
            if (onlineCompilation) {
                return {nullptr, ov::intel_npu::CompilerType::DRIVER};
            }
            OPENVINO_THROW("Failed to create plugin compiler for offline or cross compilation.");
        }

        _pluginCompilerPresence.store(PluginCompilerPresence::PRESENT, std::memory_order_release);
        std::shared_ptr<IProfilingDecoder> profilingDecoder;
        try {
            profilingDecoder = makeVCLProfilingDecoder(_vclFunctions());
        } catch (const std::exception& ex) {
            Logger::global().warning("Failed to create the VCL profiling decoder for plugin compilation: %s",
                                     ex.what());
        }

        auto pluginAdapter =
            std::make_unique<PluginCompilerAdapter>(engineBackend ? engineBackend->getInitStructs() : nullptr,
                                                    pluginCompiler,
                                                    profilingDecoder);
        return {std::move(pluginAdapter), ov::intel_npu::CompilerType::PLUGIN};
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
