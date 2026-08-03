// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// SIMD inner kernels for OSCAR (INT2) read path.
//
// K-scorer:  acc = sum_j q[j] * (code[j]*delta[sg,j] + zp[sg,j])
// V-accum:   out[j] += scale * (code[j]*delta[sg,j] + zp[sg,j])
//
// code lives in packed 2-bit stream `row[head_dim/4]`. delta/zp are fp16 arrays
// of length OSCAR_SUBGROUPS * head_dim (subgroup-major). sub_g selects which
// (delta,zp) row applies for this token.

#pragma once

#include <cstddef>
#include <cstdint>

#include "nodes/kernels/scaled_attn/codecs/codecs.hpp"
#include "nodes/kernels/simd/simd.hpp"
#include "nodes/kernels/simd/simd_loop.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov::Extensions::Cpu::XARCH {

// SIMD-decoded OSCAR 2-bit unit tile: emits `code*delta + zp` at position j.
// Only meaningful when W lanes fit in the (row, delta, zp) contiguous span
// starting at j. Caller is responsible for the loop driver + tail.
template <simd::isa I>
inline simd::f32_t<I> oscar_decode_at(const uint8_t* row,
                                      const ov::float16* delta_sg,
                                      const ov::float16* zp_sg,
                                      int j,
                                      simd::vec<int32_t, I> shifts,
                                      simd::vec<int32_t, I> mask3) {
    if constexpr (I == simd::isa::scalar) {
        (void)shifts;
        (void)mask3;
        const int code = (row[j >> 2] >> ((j & 3) * 2)) & 0x3;
        return simd::f32_t<I>(static_cast<float>(code) * static_cast<float>(delta_sg[j])
                              + static_cast<float>(zp_sg[j]));
    } else {
        // Unpack W 2-bit codes starting at element j. Requires W*2 bits = W/4 bytes
        // aligned on a byte boundary — always true when j is a multiple of W (>= 4).
        auto codes = unpack_2bit<I>(row + (j >> 2), shifts, mask3, 0);
        auto delta = simd::load<simd::f32_t<I>>(delta_sg + j);
        auto zp    = simd::load<simd::f32_t<I>>(zp_sg + j);
        return simd::fmadd(to_f32(codes), delta, zp);
    }
}

// Q·(dequant K) dot: returns sum(q[j] * (code[j]*delta[j] + zp[j])) for j in [0,dim).
inline float oscar_kdot(const float* q,
                        const uint8_t* row,
                        const ov::float16* delta_sg,
                        const ov::float16* zp_sg,
                        int dim) {
    constexpr simd::isa I = simd::active_isa;
    simd::vec<int32_t, I> shifts{};
    simd::vec<int32_t, I> mask3{};
    if constexpr (I != simd::isa::scalar) {
        alignas(64) int32_t shifts_data[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        shifts = simd::load<simd::vec<int32_t, I>>(shifts_data);
        mask3 = simd::vec<int32_t, I>(3);
    }
    return simd::simd_loop_reduce<4>(
        dim,
        [&](int j, simd::f32& acc) {
            auto k = oscar_decode_at<I>(row, delta_sg, zp_sg, j, shifts, mask3);
            acc = simd::fmadd(simd::load<simd::f32>(q + j), k, acc);
        },
        [&](int j, float& tail) {
            simd::vec<int32_t, simd::isa::scalar> zs{};
            const int code = (row[j >> 2] >> ((j & 3) * 2)) & 0x3;
            tail += q[j] * (static_cast<float>(code) * static_cast<float>(delta_sg[j])
                            + static_cast<float>(zp_sg[j]));
            (void)zs;
        });
}

// Dequantize a full OSCAR row into a f32 tile: out[j] = code[j]*delta[j] + zp[j].
inline void oscar_dequant_row(const uint8_t* row,
                              const ov::float16* delta_sg,
                              const ov::float16* zp_sg,
                              int dim,
                              float* out) {
    constexpr simd::isa I = simd::active_isa;
    simd::vec<int32_t, I> shifts{};
    simd::vec<int32_t, I> mask3{};
    if constexpr (I != simd::isa::scalar) {
        alignas(64) int32_t shifts_data[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        shifts = simd::load<simd::vec<int32_t, I>>(shifts_data);
        mask3 = simd::vec<int32_t, I>(3);
    }
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        if constexpr (Ia == simd::isa::scalar) {
            const int code = (row[j >> 2] >> ((j & 3) * 2)) & 0x3;
            out[j] = static_cast<float>(code) * static_cast<float>(delta_sg[j])
                     + static_cast<float>(zp_sg[j]);
        } else {
            auto v = oscar_decode_at<Ia>(row, delta_sg, zp_sg, j,
                                          simd::vec<int32_t, Ia>{shifts.v},
                                          simd::vec<int32_t, Ia>{mask3.v});
            simd::store(v, out + j, a);
        }
    });
}

// Plain f32 dot product (used after oscar_dequant_row).
inline float oscar_f32_dot(const float* a, const float* b, int dim) {
    return simd::simd_loop_reduce<4>(
        dim,
        [&](int j, simd::f32& acc) {
            acc = simd::fmadd(simd::load<simd::f32>(a + j), simd::load<simd::f32>(b + j), acc);
        },
        [&](int j, float& tail) { tail += a[j] * b[j]; });
}

// V weighted accumulate: acc[j] += scale * (code[j]*delta[j] + zp[j]) for j in [0,dim).
inline void oscar_vaccum(float* acc,
                         const uint8_t* row,
                         const ov::float16* delta_sg,
                         const ov::float16* zp_sg,
                         float scale,
                         int dim) {
    constexpr simd::isa I = simd::active_isa;
    simd::vec<int32_t, I> shifts{};
    simd::vec<int32_t, I> mask3{};
    if constexpr (I != simd::isa::scalar) {
        alignas(64) int32_t shifts_data[16] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        shifts = simd::load<simd::vec<int32_t, I>>(shifts_data);
        mask3 = simd::vec<int32_t, I>(3);
    }
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        if constexpr (Ia == simd::isa::scalar) {
            const int code = (row[j >> 2] >> ((j & 3) * 2)) & 0x3;
            acc[j] += scale * (static_cast<float>(code) * static_cast<float>(delta_sg[j])
                               + static_cast<float>(zp_sg[j]));
        } else {
            V vscale(scale);
            auto v_dec = oscar_decode_at<Ia>(row, delta_sg, zp_sg, j,
                                             simd::vec<int32_t, Ia>{shifts.v},
                                             simd::vec<int32_t, Ia>{mask3.v});
            simd::store(simd::fmadd(vscale, v_dec, simd::load<V>(acc + j, a)), acc + j, a);
        }
    });
}

// Q·(fp16 residual unit vector) dot.
inline float oscar_residual_dot(const float* q, const ov::float16* unit, int dim) {
    return simd::simd_loop_reduce<4>(
        dim,
        [&](int j, simd::f32& acc) {
            acc = simd::fmadd(simd::load<simd::f32>(q + j),
                              simd::load<simd::f32>(unit + j), acc);
        },
        [&](int j, float& tail) {
            tail += q[j] * static_cast<float>(unit[j]);
        });
}

// In-place scale: v[j] *= s for j in [0,dim).
inline void oscar_scale(float* v, float s, int dim) {
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        V vs(s);
        simd::store(simd::load<V>(v + j, a) * vs, v + j, a);
    });
}

// Sum of squares.
inline float oscar_sumsq(const float* v, int dim) {
    return simd::simd_loop_reduce<4>(
        dim,
        [&](int j, simd::f32& acc) {
            auto x = simd::load<simd::f32>(v + j);
            acc = simd::fmadd(x, x, acc);
        },
        [&](int j, float& tail) { tail += v[j] * v[j]; });
}

// f32 -> fp16 typed store.
inline void oscar_store_f16(const float* src, ov::float16* dst, int dim) {
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        simd::store(simd::load<V>(src + j, a), dst + j, a);
    });
}

// fp16 -> f32 typed load-store.
inline void oscar_load_f16(const ov::float16* src, float* dst, int dim) {
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        simd::store(simd::load<V>(src + j, a), dst + j, a);
    });
}

// V accumulate with f32 tile input: acc[j] += scale * tile[j].
inline void oscar_residual_vaccum_f32(float* acc, const float* tile, float scale, int dim) {
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        V vscale(scale);
        simd::store(simd::fmadd(vscale, simd::load<V>(tile + j, a),
                                simd::load<V>(acc + j, a)),
                    acc + j, a);
    });
}

// V residual accumulate: acc[j] += scale * unit[j] (fp16).
inline void oscar_residual_vaccum(float* acc, const ov::float16* unit, float scale, int dim) {
    simd::simd_loop(dim, [&](int j, auto a) {
        constexpr auto Ia = std::decay_t<decltype(a)>::isa_tag::value;
        using V = simd::f32_t<Ia>;
        V vscale(scale);
        simd::store(simd::fmadd(vscale, simd::load<V>(unit + j, a),
                                simd::load<V>(acc + j, a)),
                    acc + j, a);
    });
}

}  // namespace ov::Extensions::Cpu::XARCH
