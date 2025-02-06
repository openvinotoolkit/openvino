// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace cldnn {

float device::get_gops(cldnn::data_types dt) const {
    auto info = get_info();
    if (info.vendor_id != INTEL_VENDOR_ID) {
        // GOPS calculation is not supported for non Intel GPUs
        return 0.0f;
    }
    auto freqGHz = info.gpu_frequency / 1000.f;
    auto numEUs = info.execution_units_count;
    auto opsPerComputeBlock = 0;
    auto computeBlockIPC = 1.0f;
    switch (dt) {
    case cldnn::data_types::u8:
    case cldnn::data_types::i8: {
        if (info.supports_immad) {
            if (info.gfx_ver.major == 12) {
                if (info.gfx_ver.minor == 5)
                    opsPerComputeBlock = 512;
                else if (info.gfx_ver.minor == 7)
                    opsPerComputeBlock = 256;
            }
        } else if (info.supports_imad) {
            // fma * simd size
            opsPerComputeBlock = 2 * 32;
        } else {
            // separate mul + add instructions for int8 data type
            opsPerComputeBlock = 2 * 16;
            // mul/add instructions can't be executed in parallel, so we need 2 clocks to execute compute block
            computeBlockIPC = 0.5f;
        }
        break;
    }
    case cldnn::data_types::f16: {
        if (info.supports_immad) {
            if (info.gfx_ver.major == 12) {
                if (info.gfx_ver.minor == 5)
                    opsPerComputeBlock = 256;
                else if (info.gfx_ver.minor == 7)
                    opsPerComputeBlock = 128;
            }
        } else {
            // fma * simd size
            opsPerComputeBlock = 2 * 16;
        }
        break;
    }
    case cldnn::data_types::f32: {
        // fma * simd size
        opsPerComputeBlock = 2 * 8;
        break;
    }

    default: OPENVINO_ASSERT(false, "[GPU] get_gops: unsupported precision: ", dt);
    }

    return freqGHz * opsPerComputeBlock * computeBlockIPC * numEUs;
}

bool device::use_unified_shared_memory() const {
    GPU_DEBUG_IF(ExecutionConfig::get_disable_usm()) {
        return false;
    }
    if (get_mem_caps().supports_usm()) {
        return true;
    }
    return false;
}

}  // namespace cldnn
