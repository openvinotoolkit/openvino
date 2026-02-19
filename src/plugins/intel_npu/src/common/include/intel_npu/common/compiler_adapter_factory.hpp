// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class CompilerAdapterFactory final {
public:
    ov::intel_npu::CompilerType determinteAppropriateCompilerTypeBasedOnPlatform(std::string_view platform) const;

    std::unique_ptr<ICompilerAdapter> getCompiler(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                  ov::intel_npu::CompilerType& compilerType,
                                                  std::string_view platform) const;

private:
    inline static std::atomic<bool> _pluginCompilerIsPresent{true};
};

}  // namespace intel_npu
