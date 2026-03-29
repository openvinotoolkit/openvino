// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "codecs.hpp"
#include "polarq_tables.h"

namespace ov::Extensions::Cpu::XARCH {

// ---------------------------------------------------------------------------
// Polar codecs: fused SIMD unpack + cos/sin LUT decode for PolarQuant.
// Each codec provides:
//   void decode(const uint8_t* base, int bit_offset, simd::vec<float, i>& cos, sin) const;
//   static constexpr int bits;  // bits per element (angle)
// ---------------------------------------------------------------------------

template <simd::isa i = simd::active_isa>
struct PolarCodec5 {
    static constexpr int bits = 5;
    simd::table<32, i> cos, sin;
    explicit PolarCodec5(const PolarqLevelLUT& lut) : cos(lut.cos_lut), sin(lut.sin_lut) {}
    inline void decode(const uint8_t* base, int bit_offset, simd::vec<float, i>& c, simd::vec<float, i>& s) const {
        auto idx = unpack_5bit<i>(base, bit_offset);
        c = cos.lookup(idx);
        s = sin.lookup(idx);
    }
};

template <simd::isa i = simd::active_isa>
struct PolarCodec4 {
    static constexpr int bits = 4;
    simd::table<16, i> cos, sin;
    explicit PolarCodec4(const PolarqLevelLUT& lut) : cos(lut.cos_lut), sin(lut.sin_lut) {}
    inline void decode(const uint8_t* base, int bit_offset, simd::vec<float, i>& c, simd::vec<float, i>& s) const {
        auto idx = unpack_4bit<i>(base, bit_offset);
        c = cos.lookup(idx);
        s = sin.lookup(idx);
    }
};

template <simd::isa i = simd::active_isa>
struct PolarCodec3 {
    static constexpr int bits = 3;
    static constexpr int32_t shifts_data[] = {0, 3, 6, 9, 12, 15, 18, 21, 0, 3, 6, 9, 12, 15, 18, 21};
    simd::table<8, i> cos, sin;
    simd::vec<int32_t, i> shifts, mask;
    explicit PolarCodec3(const PolarqLevelLUT& lut)
        : cos(lut.cos_lut),
          sin(lut.sin_lut),
          shifts(simd::load<simd::vec<int32_t, i>>(shifts_data)),
          mask(7) {}
    inline void decode(const uint8_t* base, int bit_offset, simd::vec<float, i>& c, simd::vec<float, i>& s) const {
        auto idx = unpack_3bit<i>(base, shifts, mask, bit_offset);
        c = cos.lookup(idx);
        s = sin.lookup(idx);
    }
};

template <simd::isa i = simd::active_isa>
struct PolarCodec2 {
    static constexpr int bits = 2;
    static constexpr int32_t shifts_data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    simd::table<4, i> cos, sin;
    simd::vec<int32_t, i> shifts, mask;
    explicit PolarCodec2(const PolarqLevelLUT& lut)
        : cos(lut.cos_lut),
          sin(lut.sin_lut),
          shifts(simd::load<simd::vec<int32_t, i>>(shifts_data)),
          mask(3) {}
    inline void decode(const uint8_t* base, int bit_offset, simd::vec<float, i>& c, simd::vec<float, i>& s) const {
        auto idx = unpack_2bit<i>(base, shifts, mask, bit_offset);
        c = cos.lookup(idx);
        s = sin.lookup(idx);
    }
};

}  // namespace ov::Extensions::Cpu::XARCH
