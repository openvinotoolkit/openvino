// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Common declarations for the simd namespace: isa enum, active_isa,
// primary templates for vec and mask, and utility constexpr functions.
// No ISA-specific code — included by all per-ISA headers.

#pragma once

#include <cstdint>

namespace ov::Extensions::Cpu::XARCH::simd {

enum class isa : uint8_t { scalar, avx2, avx512 };

#if defined(HAVE_AVX512F)
inline constexpr isa active_isa = isa::avx512;
#elif defined(HAVE_AVX2)
inline constexpr isa active_isa = isa::avx2;
#else
inline constexpr isa active_isa = isa::scalar;
#endif

constexpr int shuffle_ctrl(int d3, int d2, int d1, int d0) {
    return (d3 << 6) | (d2 << 4) | (d1 << 2) | d0;
}

constexpr int lane_ctrl(int out0, int out1) {
    return out0 | (out1 << 4);
}

template <typename T = float, isa i = active_isa>
struct vec;

template <isa i>
struct mask;

}  // namespace ov::Extensions::Cpu::XARCH::simd
