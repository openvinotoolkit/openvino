// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "nodes/kernels/simd/simd.hpp"

namespace ov::Extensions::Cpu {

// Random sign vector (cached, ±1.0f, dim elements) for MSE WHT rotation.
const float* turboq_get_wht_signs(int dim);

namespace XARCH {

// ---------------------------------------------------------------------------
// Walsh-Hadamard transform (WHT) — in-place, iterative butterfly.
// ---------------------------------------------------------------------------
inline void turboq_wht_inplace(float* data, int dim) {
    for (int h = 1; h < dim; h <<= 1) {
        const int stride = h << 1;
        if (h >= simd::f32::width) {
            for (int base = 0; base < dim; base += stride) {
                for (int j = 0; j < h; j += simd::f32::width) {
                    float* lo = data + base + j;
                    float* hi = data + base + j + h;
                    auto a = simd::load<simd::f32>(lo);
                    auto b = simd::load<simd::f32>(hi);
                    store(a + b, lo);
                    store(a - b, hi);
                }
            }
        } else {
            for (int base = 0; base < dim; base += stride) {
                for (int j = 0; j < h; j++) {
                    float* lo = data + base + j;
                    float* hi = data + base + j + h;
                    float a = *lo;
                    float b = *hi;
                    *lo = a + b;
                    *hi = a - b;
                }
            }
        }
    }
}

// Forward: y = (H * diag(signs) * x) / sqrt(dim)
// Equivalent to a randomized Hadamard rotation (orthogonal transform).
inline void turboq_wht_forward(const float* signs, const float* x, float* y, int dim) {
    constexpr int W = simd::f32::width;
    const float inv_sqrt_dim = 1.0F / std::sqrt(static_cast<float>(dim));
    // Step 1: element-wise multiply by signs
    for (int i = 0; i + W - 1 < dim; i += W) {
        store(simd::load<simd::f32>(x + i) * simd::load<simd::f32>(signs + i), y + i);
    }
    // Step 2: in-place WHT
    turboq_wht_inplace(y, dim);
    // Step 3: normalize by 1/sqrt(dim)
    auto vscale = simd::f32(inv_sqrt_dim);
    for (int i = 0; i + W - 1 < dim; i += W) {
        store(simd::load<simd::f32>(y + i) * vscale, y + i);
    }
}

// Inverse: x = diag(signs) * H * y / sqrt(dim)
// WHT runs in-place on y (caller-owned f32 workspace), final scale+signs writes to typed x.
template <typename T>
inline void turboq_wht_inverse(const float* signs, float* y, T* x, int dim) {
    constexpr int W = simd::f32::width;
    const float inv_sqrt_dim = 1.0F / std::sqrt(static_cast<float>(dim));
    turboq_wht_inplace(y, dim);
    auto vscale = simd::f32(inv_sqrt_dim);
    for (int i = 0; i + W - 1 < dim; i += W) {
        store(simd::load<simd::f32>(y + i) * vscale * simd::load<simd::f32>(signs + i), x + i);
    }
}

}  // namespace XARCH
}  // namespace ov::Extensions::Cpu
