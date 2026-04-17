// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared encode helpers: L2 norm + unit-vector normalization,
// branchless scalar quantize via boundary linear-scan.
// These are used by TurboQ, TurboQ-QJL, and PolarQuant encoders.
//
// Header-only: included by per-codec `*_quantize.hpp` headers, which in turn
// are included by the cross-compiled mha_kv_cache_codec.cpp TU, so SIMD code
// compiles for the active ISA without requiring a separate TU.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "nodes/kernels/simd/simd.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Compute L2 norm and normalize src to unit vector in f32. Returns the norm.
// Dispatches on src_precision to handle bf16/f16→f32 conversion during the load.
template <typename T>
static float normalize_to_unit(const void* src_raw, float* unit, int dim) {
    const auto* src = static_cast<const T*>(src_raw);

    float norm_sq = simd::simd_loop_reduce<2>(
        dim,
        [&](int i, simd::f32& acc) {
            auto v = simd::load<simd::f32>(src + i);
            acc = simd::fmadd(v, v, acc);
        },
        [&](int i, float& tail) {
            auto v = static_cast<float>(src[i]);
            tail += v * v;
        });
    const float norm = std::sqrt(norm_sq);

    if (norm < 1e-30F) {
        std::memset(unit, 0, dim * sizeof(float));
    } else {
        const float inv_norm = 1.0F / norm;
        simd::simd_loop(dim, [&](int i, auto a) {
            using V = simd::vec<float, decltype(a)::isa_tag::value>;
            simd::store(simd::load<V>(src + i, a) * V(inv_norm), unit + i, a);
        });
    }
    return norm;
}

static inline float dispatch_normalize(const void* src, float* unit, int dim, ov::element::Type precision) {
    if (precision == ov::element::bf16) {
        return normalize_to_unit<ov::bfloat16>(src, unit, dim);
    }
    if (precision == ov::element::f16) {
        return normalize_to_unit<ov::float16>(src, unit, dim);
    }
    return normalize_to_unit<float>(src, unit, dim);
}

// Branchless linear scan over boundaries — shared by TBQ / QJL / Polar.
static inline uint8_t scalar_quantize(float x, const float* boundaries, int n_boundaries) {
    int idx = 0;
    for (int i = 0; i < n_boundaries; i++) {
        idx += (x > boundaries[i]) ? 1 : 0;
    }
    return static_cast<uint8_t>(idx);
}

}  // namespace ov::Extensions::Cpu::XARCH
