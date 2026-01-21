// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  ov::intel_npu::CompilerType compilerType,
                                                  ov::AnyMap& properties) const {
        if (_pluginCompilerIsPresent) {
            if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                try {
                    properties[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
                    return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
                } catch (...) {
                    _pluginCompilerIsPresent = false;
                    properties[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::DRIVER;
                    compilerType = ov::intel_npu::CompilerType::DRIVER;
                }
            }
        } else {
            properties[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::DRIVER;
            compilerType = ov::intel_npu::CompilerType::DRIVER;
        }

        if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
            if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
                return std::make_unique<PluginCompilerAdapter>(nullptr);
            }

            return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
        } else if (compilerType == ov::intel_npu::CompilerType::DRIVER) {
            if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
                OPENVINO_THROW("NPU Compiler Adapter must be used with LEVEL0 backend");
            }

            return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs());
        } else {
            OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
        }
    }

private:
    mutable std::atomic<bool> _pluginCompilerIsPresent = true;
};

}  // namespace intel_npu
