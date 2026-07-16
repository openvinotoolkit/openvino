// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "codecs.hpp"
#include "turboq_tables.hpp"

namespace ov::Extensions::Cpu::XARCH {

// ---------------------------------------------------------------------------
// TurboQ decoders: SIMD codebook decode for 4-bit, 3-bit, 2-bit quantization.
//
// Stateful (hold per-ISA codebook + shifts + mask) to match the stateless
// Decoder interface used by codec_kernels. decode() takes a Params (ignored)
// and simd::active_lanes<I> context. When called with scalar active_lanes
// (e.g. from codec_dot tail), falls back to scalar codebook lookup.
// ---------------------------------------------------------------------------

template <simd::isa I = simd::active_isa>
struct TurboQDecoder4 {
    static constexpr int bits = 4;
    const float* cb_raw;
    simd::table<16, I> cb;

    explicit TurboQDecoder4(int head_dim = 0) : cb_raw(turboq_codebook(4, head_dim)), cb(cb_raw) {}

    template <simd::isa Ia, typename Params>
    simd::f32_t<Ia> decode(const uint8_t* base,
                           int bit_offset,
                           const Params& /*p*/,
                           simd::active_lanes<Ia> /*unused*/) const {
        if constexpr (Ia == simd::isa::scalar) {
            int idx = (*base >> bit_offset) & 0x0F;
            return simd::f32_t<Ia>(cb_raw[idx]);
        } else {
            return cb.lookup(unpack_4bit<Ia>(base, bit_offset));
        }
    }
};

template <simd::isa I = simd::active_isa>
struct TurboQDecoder3 {
    static constexpr int bits = 3;
    static constexpr int32_t shifts_data[] = {0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21};
    const float* cb_raw;
    simd::table<8, I> cb;
    simd::vec<int32_t, I> shifts;
    simd::vec<int32_t, I> mask;

    explicit TurboQDecoder3(int head_dim = 0)
        : cb_raw(turboq_codebook(3, head_dim)),
          cb(cb_raw),
          shifts(simd::load<simd::vec<int32_t, I>>(shifts_data)),
          mask(7) {}

    template <simd::isa Ia, typename Params>
    simd::f32_t<Ia> decode(const uint8_t* base,
                           int bit_offset,
                           const Params& /*p*/,
                           simd::active_lanes<Ia> /*unused*/) const {
        if constexpr (Ia == simd::isa::scalar) {
            auto w = read_as<uint32_t>(base);
            auto idx = static_cast<int>((w >> bit_offset) & 0x7);
            return simd::f32_t<Ia>(cb_raw[idx]);
        } else {
            return cb.lookup(unpack_3bit<Ia>(base, shifts, mask, bit_offset));
        }
    }
};

template <simd::isa I = simd::active_isa>
struct TurboQDecoder2 {
    static constexpr int bits = 2;
    static constexpr int32_t shifts_data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    const float* cb_raw;
    simd::table<4, I> cb;
    simd::vec<int32_t, I> shifts;
    simd::vec<int32_t, I> mask;

    explicit TurboQDecoder2(int head_dim = 0)
        : cb_raw(turboq_codebook(2, head_dim)),
          cb(cb_raw),
          shifts(simd::load<simd::vec<int32_t, I>>(shifts_data)),
          mask(3) {}

    template <simd::isa Ia, typename Params>
    simd::f32_t<Ia> decode(const uint8_t* base,
                           int bit_offset,
                           const Params& /*p*/,
                           simd::active_lanes<Ia> /*unused*/) const {
        if constexpr (Ia == simd::isa::scalar) {
            auto w = read_as<uint32_t>(base);
            auto idx = static_cast<int>((w >> bit_offset) & 0x3);
            return simd::f32_t<Ia>(cb_raw[idx]);
        } else {
            return cb.lookup(unpack_2bit<Ia>(base, shifts, mask, bit_offset));
        }
    }
};

// Packed byte size for a full KV row (all heads for one token).
// turboq_head_bytes is declared in mha_kv_cache_codec.hpp.
size_t turboq_row_bytes(int num_kv_heads, int head_dim, int bits);

}  // namespace ov::Extensions::Cpu::XARCH
