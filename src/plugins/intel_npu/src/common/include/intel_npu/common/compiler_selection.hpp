// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "intel_npu/npu_private_properties.hpp"

namespace intel_npu {

/**
 * @brief Determines whether the compiler-in-plugin should be preferred for a canonical platform.
 *
 * The 3720 threshold is specific to Windows driver versions; Linux uses a different version encoding.
 */
inline bool isPluginCompilerPreferred(const std::string_view platform,
                                      const std::optional<uint32_t> driverVersion,
                                      const bool windowsDriver) {
    if (platform == ov::intel_npu::Platform::NPU4000 || platform == ov::intel_npu::Platform::NPU5010 ||
        platform == ov::intel_npu::Platform::NPU5020 || platform == ov::intel_npu::Platform::NPU6010) {
        return true;
    }

    if (platform != ov::intel_npu::Platform::NPU3720) {
        return false;
    }

    if (!windowsDriver) {
        return true;
    }

    constexpr uint32_t minimumWindowsDriverVersion = 2565;
    return driverVersion.has_value() && driverVersion.value() >= minimumWindowsDriverVersion;
}

}  // namespace intel_npu
