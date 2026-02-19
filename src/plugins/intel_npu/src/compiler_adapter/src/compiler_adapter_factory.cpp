// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include "driver_compiler_adapter.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

ov::intel_npu::CompilerType CompilerAdapterFactory::determinteAppropriateCompilerTypeBasedOnPlatform(
    std::string_view platform) const {
    if (platform == ov::intel_npu::Platform::NPU4000 || platform == ov::intel_npu::Platform::NPU5010) {
        return ov::intel_npu::CompilerType::PLUGIN;
    }

    return ov::intel_npu::CompilerType::DRIVER;
}

std::unique_ptr<ICompilerAdapter> CompilerAdapterFactory::getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                                      ov::intel_npu::CompilerType& compilerType,
                                                                      std::string_view platform) const {
    if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        if (engineBackend != nullptr) {
            compilerType = determinteAppropriateCompilerTypeBasedOnPlatform(platform);
            if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
                if (_pluginCompilerIsPresent) {
                    try {
                        return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
                    } catch (...) {
                        _pluginCompilerIsPresent = false;
                        compilerType = ov::intel_npu::CompilerType::DRIVER;
                    }
                } else {
                    // plugin compiler isn't present, fallback to driver compiler
                    compilerType = ov::intel_npu::CompilerType::DRIVER;
                }
            }
        } else {
            // device isn't available, offline compilation only
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
        }
    }

    if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
        if (engineBackend == nullptr) {
            return std::make_unique<PluginCompilerAdapter>(nullptr);
        }

        return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
    } else if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
        if (engineBackend == nullptr || engineBackend->getDevice() == nullptr) {
            OPENVINO_THROW(
                "Could not find a NPU device. Using driver compiler requires a valid device to be present in "
                "the system.");
        }

        // It is required to check if the device is compatible with the provided platform, as the driver compiler
        // will be used.
        auto deviceName = engineBackend->getDevice()->getName();
        if (deviceName != platform && deviceName != "AUTO_DETECT") {
            OPENVINO_THROW("Could not find a valid NPU device for the provided configuration.");
        }

        return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs());
    } else {
        OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
    }
}

}  // namespace intel_npu
