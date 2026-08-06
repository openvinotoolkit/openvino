// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

bool isCompilerOptionSupported(ov::intel_npu::CompilerType compilerType,
                               const std::string& optionName,
                               const std::optional<std::string>& optionValue = std::nullopt,
                               const ov::SoPtr<IEngineBackend>& engineBackend = ov::SoPtr<IEngineBackend>{},
                               const std::optional<uint32_t>& compilerSupportVersion = std::nullopt);

std::optional<std::vector<std::string>> getCompilerSupportedOptions(
    ov::intel_npu::CompilerType compilerType,
    const ov::SoPtr<IEngineBackend>& engineBackend = ov::SoPtr<IEngineBackend>{});

}  // namespace intel_npu
