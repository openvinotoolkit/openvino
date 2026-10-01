// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/runtime/device_info.hpp"
#ifdef ENABLE_DEBUG_CAPS
#    include <cstdlib>
#    include <iostream>
#    include <stdexcept>
#endif

namespace cldnn {

/// @brief Debug builds only: OV_GPU_ARCH_OVERRIDE=<arch> makes the device report another arch, so the host code of a
/// generation can be exercised (its kernel sources dumped) on a device of a different one. Kernels built for the forged
/// arch do NOT run on the real device: dump only, never read test results from such a run. Anything else than a known name
/// throws, so a typo is not silently ignored. Release builds return `detected`.
inline gpu_arch debug_arch_override(gpu_arch detected) {
#ifdef ENABLE_DEBUG_CAPS
    const char* env = std::getenv("OV_GPU_ARCH_OVERRIDE");
    if (env == nullptr || env[0] == '\0')
        return detected;
    static const struct {
        const char* name;
        gpu_arch arch;
    } names[] = {{"gen9", gpu_arch::gen9},
                 {"gen11", gpu_arch::gen11},
                 {"xe_lp", gpu_arch::xe_lp},
                 {"xe_hp", gpu_arch::xe_hp},
                 {"xe_hpg", gpu_arch::xe_hpg},
                 {"xe_hpc", gpu_arch::xe_hpc},
                 {"xe2", gpu_arch::xe2},
                 {"xe3", gpu_arch::xe3},
                 {"xe3p", gpu_arch::xe3p}};
    for (const auto& n : names) {
        if (std::string(env) == n.name) {
            if (n.arch != detected) {
                static bool announced = false;
                if (!announced) {
                    announced = true;
                    std::cerr << "[GPU] OV_GPU_ARCH_OVERRIDE: arch " << static_cast<int>(detected) << " -> " << env
                              << " (debug only, kernels built for " << env << " will not run on this device)" << std::endl;
                }
            }
            return n.arch;
        }
    }
    throw std::runtime_error(std::string("[GPU] OV_GPU_ARCH_OVERRIDE: unknown arch '") + env + "'");
#else
    return detected;
#endif
}

}  // namespace cldnn
