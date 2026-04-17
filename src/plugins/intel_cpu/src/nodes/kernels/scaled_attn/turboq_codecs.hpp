// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "codecs.hpp"
#include "turboq_tables.h"

namespace ov::Extensions::Cpu::XARCH {

// ---------------------------------------------------------------------------
// TurboQ codecs: SIMD codebook decode for 4-bit, 3-bit, 2-bit quantization.
// Each codec provides:
//   simd::vec<float, i> decode(const uint8_t* base, int bit_offset) const;
//   static constexpr int bits;  // bits per element
// ---------------------------------------------------------------------------

template <simd::isa i = simd::active_isa>
struct TurboQCodec4 {
    static constexpr int bits = 4;
    simd::table<16, i> cb;
    explicit TurboQCodec4(int head_dim = 0) : cb(turboq_codebook(4, head_dim)) {}
    inline simd::vec<float, i> decode(const uint8_t* base, int bit_offset) const {
        return cb.lookup(unpack_4bit<i>(base, bit_offset));
    }
};

template <simd::isa i = simd::active_isa>
struct TurboQCodec3 {
    static constexpr int bits = 3;
    static constexpr int32_t shifts_data[] = {0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21};
    simd::table<8, i> cb;
    simd::vec<int32_t, i> shifts;
    simd::vec<int32_t, i> mask;
    explicit TurboQCodec3(int head_dim = 0)
        : cb(turboq_codebook(3, head_dim)),
          shifts(simd::load<simd::vec<int32_t, i>>(shifts_data)),
          mask(7) {}
    inline simd::vec<float, i> decode(const uint8_t* base, int bit_offset) const {
        return cb.lookup(unpack_3bit<i>(base, shifts, mask, bit_offset));
    }
};

template <simd::isa i = simd::active_isa>
struct TurboQCodec2 {
    static constexpr int bits = 2;
    static constexpr int32_t shifts_data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    simd::table<4, i> cb;
    simd::vec<int32_t, i> shifts;
    simd::vec<int32_t, i> mask;
    explicit TurboQCodec2(int head_dim = 0)
        : cb(turboq_codebook(2, head_dim)),
          shifts(simd::load<simd::vec<int32_t, i>>(shifts_data)),
          mask(3) {}
    inline simd::vec<float, i> decode(const uint8_t* base, int bit_offset) const {
        return cb.lookup(unpack_2bit<i>(base, shifts, mask, bit_offset));
    }
};

}  // namespace ov::Extensions::Cpu::XARCH
