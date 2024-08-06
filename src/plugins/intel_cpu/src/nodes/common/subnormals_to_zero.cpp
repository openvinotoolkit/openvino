// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subnormals_to_zero.h"

#include <cstddef>
#include <cstdint>

namespace ov {
namespace intel_cpu {

// @todo add optimized implementation as Eltwise / emitter
void setSubnormalsToZero(float* data, size_t size) {
    uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        if ((u32data[i] & (0xFF << 23)) == 0) {
            u32data[i] = 0;
        }
    }
}

}   // namespace intel_cpu
}   // namespace ov
