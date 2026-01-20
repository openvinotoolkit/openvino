// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

struct CompilerTypeSettings {
    CompilerTypeSettings() = default;
    CompilerTypeSettings(bool isSet, ov::intel_npu::CompilerType type)
        : compilerConfigIsSet(isSet),
          compilerType(type) {}

    bool compilerConfigIsSet = false;
    ov::intel_npu::CompilerType compilerType;
};

class CompilerAdapterFactory final {
public:
    std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  const CompilerTypeSettings& compilerTypeSettings,
                                                  FilteredConfig& globalConfig) const {
        auto compilerType = compilerTypeSettings.compilerType;

        if (!compilerTypeSettings.compilerConfigIsSet) {
            try {
                if (compilerType == ov::intel_npu::CompilerType::PLUGIN) {
                    auto compiler = std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
                    return compiler;
                }
            } catch (...) {
                globalConfig.update({{ov::intel_npu::compiler_type.name(), "DRIVER"}});
                compilerType = ov::intel_npu::CompilerType::DRIVER;
            }
        }

        switch (compilerType) {
        case ov::intel_npu::CompilerType::PLUGIN: {
            if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
                return std::make_unique<PluginCompilerAdapter>(nullptr);
            }
            return std::make_unique<PluginCompilerAdapter>(engineBackend->getInitStructs());
        }
        case ov::intel_npu::CompilerType::DRIVER: {
            if (engineBackend == nullptr || engineBackend->getName() != "LEVEL0") {
                OPENVINO_THROW("NPU Compiler Adapter must be used with LEVEL0 backend");
            }

            return std::make_unique<DriverCompilerAdapter>(engineBackend->getInitStructs());
        }
        default:
            OPENVINO_THROW("Invalid NPU_COMPILER_TYPE");
        }
    }
};

}  // namespace intel_npu
