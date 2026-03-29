// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>

#include "nodes/kernels/simd/simd.hpp"

namespace ov::Extensions::Cpu {

// Matrix getters — cached per dim, thread-safe. First call for a given dim allocates.
const float* turboq_get_rotation_matrix(int dim);      // Q: Haar orthogonal
const float* turboq_get_rotation_matrix_t(int dim);    // Q^T
const float* turboq_get_projection_matrix(int dim);    // S: raw Gaussian (QJL)
const float* turboq_get_projection_matrix_t(int dim);  // S^T
const float* turboq_get_SQ_matrix(int dim);            // S * Q (fused Q projection)

// Random sign vectors (cached, ±1.0f, dim elements).
const float* turboq_get_wht_signs(int dim);      // MSE WHT rotation signs
const float* turboq_get_qjl_wht_signs(int dim);  // Independent QJL WHT signs

// Rotation mode — controls which transform is used by rotate_forward/inverse.
// Set via environment variable OV_TURBOQ_ROTATION ("wht" | "dense" | "none"),
// read once on first access. Default: WHT.
enum class TurboqRotationMode : uint8_t { WHT, DENSE, NONE };
TurboqRotationMode turboq_get_rotation_mode();

// QJL projection mode — controls which transform is used for QJL sign projection.
// Set via environment variable OV_TURBOQ_QJL_PROJECTION ("wht" | "dense" | "none").
// WHT: uses independent WHT signs (O(d log d), recommended).
// DENSE: uses raw Gaussian S matrix (O(d²), original implementation).
TurboqRotationMode turboq_get_qjl_projection_mode();

namespace XARCH {

// Reference scalar matvec (used by tests and non-cross-compiled code).
void turboq_matvec_ref(const float* M, const float* x, float* y, int dim);

// Dispatching matvec: y = M * x, where M is dim x dim (row-major).
// Template on output type: final store converts f32 to T.
template <typename T = float>  // default needed: called without explicit T from forward rotation
inline void turboq_matvec(const float* M, const float* x, T* y, int dim) {
    constexpr int W = simd::f32::width;
    for (int i = 0; i < dim; i++) {
        const float* row = M + i * dim;
        simd::f32 vsum0, vsum1, vsum2, vsum3;
        int j = 0;
        for (; j + 4 * W - 1 < dim; j += 4 * W) {
            vsum0 = fmadd(simd::load<simd::f32>(row + j), simd::load<simd::f32>(x + j), vsum0);
            vsum1 = fmadd(simd::load<simd::f32>(row + j + W), simd::load<simd::f32>(x + j + W), vsum1);
            vsum2 = fmadd(simd::load<simd::f32>(row + j + 2 * W), simd::load<simd::f32>(x + j + 2 * W), vsum2);
            vsum3 = fmadd(simd::load<simd::f32>(row + j + 3 * W), simd::load<simd::f32>(x + j + 3 * W), vsum3);
        }
        for (; j + W - 1 < dim; j += W) {
            vsum0 = fmadd(simd::load<simd::f32>(row + j), simd::load<simd::f32>(x + j), vsum0);
        }
        y[i] = static_cast<T>(reduce((vsum0 + vsum1) + (vsum2 + vsum3)));
    }
}

// ---------------------------------------------------------------------------
// Dispatching rotation: forward and inverse, based on turboq_get_rotation_mode().
// These are the functions production code should call.
// ---------------------------------------------------------------------------
inline void turboq_rotate_forward(const float* src, float* dst, int dim);  // defined after WHT
template <typename T = float>
inline void turboq_rotate_inverse(float* src, T* dst, int dim);  // defined after WHT

// ---------------------------------------------------------------------------
// Walsh-Hadamard Transform + random sign flips.
// Randomized Hadamard rotation: y = H * diag(signs) * x / sqrt(dim)
// where H is the unnormalized Hadamard matrix (butterfly structure).
// O(n log n) versus O(n²) for dense matvec.
// ---------------------------------------------------------------------------

// In-place unnormalized WHT (sequency-ordered, iterative butterfly).
inline void turboq_wht_inplace(float* data, int dim) {
    constexpr int W = simd::f32::width;
    for (int half = 1; half < dim; half <<= 1) {
        if (half >= W) {
            for (int i = 0; i < dim; i += half * 2) {
                for (int j = 0; j < half; j += W) {
                    float* lo = data + i + j;
                    float* hi = lo + half;
                    auto a = simd::load<simd::f32>(lo);
                    auto b = simd::load<simd::f32>(hi);
                    store(a + b, lo);
                    store(a - b, hi);
                }
            }
        } else {
            for (int i = 0; i < dim; i += half * 2) {
                for (int j = 0; j < half; j++) {
                    float* lo = data + i + j;
                    float* hi = lo + half;
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

// ---------------------------------------------------------------------------
// Dispatching rotation implementations.
// ---------------------------------------------------------------------------
inline void turboq_rotate_forward(const float* src, float* dst, int dim) {
    switch (turboq_get_rotation_mode()) {
    case TurboqRotationMode::WHT:
        turboq_wht_forward(turboq_get_wht_signs(dim), src, dst, dim);
        break;
    case TurboqRotationMode::DENSE:
        turboq_matvec(turboq_get_rotation_matrix(dim), src, dst, dim);
        break;
    case TurboqRotationMode::NONE:
        if (src != dst) {
            std::memcpy(dst, src, dim * sizeof(float));
        }
        break;
    }
}

template <typename T>
inline void turboq_rotate_inverse(float* src, T* dst, int dim) {
    constexpr int W = simd::f32::width;
    switch (turboq_get_rotation_mode()) {
    case TurboqRotationMode::WHT:
        turboq_wht_inverse(turboq_get_wht_signs(dim), src, dst, dim);
        break;
    case TurboqRotationMode::DENSE:
        turboq_matvec(turboq_get_rotation_matrix_t(dim), src, dst, dim);
        break;
    case TurboqRotationMode::NONE:
        for (int i = 0; i + W - 1 < dim; i += W) {
            store(simd::load<simd::f32>(src + i), dst + i);
        }
        break;
    }
}

// ---------------------------------------------------------------------------
// QJL projection: forward (S * x) and inverse (S^T * x).
// Dispatches based on turboq_get_qjl_projection_mode().
// WHT mode uses independent signs (O(d log d)); DENSE uses Gaussian S (O(d²)).
// ---------------------------------------------------------------------------
inline void turboq_qjl_project_forward(const float* src, float* dst, int dim) {
    switch (turboq_get_qjl_projection_mode()) {
    case TurboqRotationMode::WHT:
        turboq_wht_forward(turboq_get_qjl_wht_signs(dim), src, dst, dim);
        break;
    case TurboqRotationMode::DENSE:
        turboq_matvec(turboq_get_projection_matrix(dim), src, dst, dim);
        break;
    case TurboqRotationMode::NONE:
        if (src != dst) {
            std::memcpy(dst, src, dim * sizeof(float));
        }
        break;
    }
}

inline void turboq_qjl_project_inverse(float* src, float* dst, int dim) {
    switch (turboq_get_qjl_projection_mode()) {
    case TurboqRotationMode::WHT:
        // WHT runs in-place on src (caller-owned workspace), scale+signs writes to dst.
        turboq_wht_inverse(turboq_get_qjl_wht_signs(dim), src, dst, dim);
        break;
    case TurboqRotationMode::DENSE:
        turboq_matvec(turboq_get_projection_matrix_t(dim), src, dst, dim);
        break;
    case TurboqRotationMode::NONE:
        if (src != dst) {
            std::memcpy(dst, src, dim * sizeof(float));
        }
        break;
    }
}

}  // namespace XARCH
}  // namespace ov::Extensions::Cpu
