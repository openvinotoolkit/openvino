// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "driver_compiler_adapter.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/config/config.hpp"
#include "plugin_compiler_adapter.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    const std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                        const Config& config) const {
        auto compilerType = config.get<COMPILER_TYPE>();
        switch (compilerType) {
        case ov::intel_npu::CompilerType::MLIR: {
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
