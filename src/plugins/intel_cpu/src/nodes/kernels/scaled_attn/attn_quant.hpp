// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cfloat>

namespace ov {
namespace intel_cpu {

inline void attn_quant_u8(uint8_t* a, float* b, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    for (; i < n; i++) {
        float tmp = b[i];
        max = std::max(max, tmp);
        min = std::min(min, tmp);
    }
    scale = (max - min) / 255;
    zp = -min / scale;

    i = 0;
    for (; i < n; i++) {
        float tmp = b[i];
        a[i] = static_cast<uint8_t>(std::round(tmp / scale + zp));
    }
}

inline void attn_dequant_u8(uint8_t* a, float* b, size_t n, float scale, float zp) {
    for (size_t i = 0; i < n; ++i) {
        float tmp = a[i];
        tmp = (tmp - zp) * scale;
        b[i] = tmp;
    }
}

}  // namespace intel_cpu
}  // namespace ov