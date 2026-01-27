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
    static CompilerAdapterFactory& getInstance() {
        static CompilerAdapterFactory instance;
        return instance;
    }

    CompilerAdapterFactory(const CompilerAdapterFactory&) = delete;
    CompilerAdapterFactory(CompilerAdapterFactory&&) = delete;
    CompilerAdapterFactory& operator=(const CompilerAdapterFactory&) = delete;
    CompilerAdapterFactory& operator=(CompilerAdapterFactory&&) = delete;

    std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  ov::intel_npu::CompilerType& compilerType) const {
        if (engineBackend) {
            if (_pluginCompilerIsPresent) {
                if (compilerType == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
                    try {
                        compilerType = ov::intel_npu::CompilerType::PLUGIN;
                        return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
                    } catch (...) {
                        _pluginCompilerIsPresent = false;
                        compilerType = ov::intel_npu::CompilerType::DRIVER;
                    }
                }
            } else {
                compilerType = ov::intel_npu::CompilerType::DRIVER;
            }
        } else {
            // no backend present, offline compilation only
            compilerType = ov::intel_npu::CompilerType::PLUGIN;
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
    CompilerAdapterFactory() = default;
    ~CompilerAdapterFactory() = default;

    mutable std::atomic<bool> _pluginCompilerIsPresent = true;
};

}  // namespace intel_npu
