// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// simd::vec<T, i> — compile-time abstraction over AVX-512 / AVX2 / scalar.
//
// vec<T, i> wraps the native register. Operators (+, -, *, &) are members.
// Everything else is a free function: load, store, reduce, fmadd, select,
// permute, shuffle, etc. — found via ADL from vec/mask arguments.
//
// File layout:
//   simd_common.hpp   — isa enum, active_isa, primary templates
//   simd_scalar.hpp   — scalar specializations (always available)
//   simd_avx2.hpp     — AVX2 specializations (no #ifdef inside)
//   simd_avx512.hpp   — AVX-512 specializations (no #ifdef inside)
//   simd.hpp (this)   — aggregator: includes above + aliases, load API, table

#pragma once

#include "simd_common.hpp"
#include "simd_scalar.hpp"
#if defined(HAVE_AVX2)
#    include "simd_avx2.hpp"
#endif
#if defined(HAVE_AVX512F)
#    include "simd_avx512.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH::simd {

// ===== Convenience aliases ================================================

using f32 = vec<float>;
using i32 = vec<int32_t>;

// ===== Public load API ====================================================

template <typename V, typename SrcT>
inline V load(const SrcT* ptr) {
    return load(ptr, static_cast<V*>(nullptr));
}

template <typename V>
inline V partial_load(uint32_t k, const typename V::element_type* ptr) {
    return partial_load(k, ptr, static_cast<V*>(nullptr));
}

template <typename V>
inline V load_u4(const uint8_t* ptr, int bit_offset) {
    return load_u4(ptr, bit_offset, static_cast<V*>(nullptr));
}

// ===== table: fixed-size SIMD look-up table ===============================

template <int N, isa i = active_isa>
struct table {
    using V = vec<float, i>;
    using VI = vec<int32_t, i>;
    static constexpr int W = V::width;
    static constexpr int n_regs = (N + W - 1) / W;
    static_assert(n_regs <= 4, "table supports up to 4 * vec::width entries");
    V regs[n_regs];

    explicit table(const float* data) {
        for (int r = 0; r < n_regs; r++) {
            int remaining = N - r * W;
            if (remaining >= W) {
                regs[r] = load<V>(data + r * W);
            } else {
                regs[r] = partial_load<V>((1U << remaining) - 1, data + r * W);
            }
        }
    }

    [[nodiscard]] V lookup(VI idx) const {
        if constexpr (n_regs == 1) {
            return permute(regs[0], idx);
        } else if constexpr (n_regs == 2 && i == isa::avx512) {
            return permute2(regs[0], idx, regs[1]);
        } else {
            auto lane = idx & VI{W - 1};
            auto res = select(idx > VI{W - 1}, permute(regs[0], lane), permute(regs[1], lane));
            if constexpr (n_regs >= 3) {
                res = select(idx > VI{2 * W - 1}, res, permute(regs[2], lane));
            }
            if constexpr (n_regs >= 4) {
                res = select(idx > VI{3 * W - 1}, res, permute(regs[3], lane));
            }
            return res;
        }
    }
};

template <int N>
struct table<N, isa::scalar> {
    const float* data;
    explicit table(const float* p) : data(p) {}
    [[nodiscard]] vec<float, isa::scalar> lookup(vec<int32_t, isa::scalar> idx) const {
        return {data[idx.v]};
    }
};

}  // namespace ov::Extensions::Cpu::XARCH::simd
