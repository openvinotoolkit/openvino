// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <type_traits>

#include "openvino/reference/convert.hpp"

namespace ov {
namespace reference {

namespace cpu {
typedef enum {
    isa_any,
    sse42,
    avx,
    avx2,
    avx512_common,
    avx512_core,
    avx512_core_vnni,
    avx512_mic,
    avx512_mic_4ops,
    avx512_core_bf16,
    avx512_vpopcnt,
    fp16
} isa_t;

bool may_i_use(const isa_t cpu_isa);
}  // namespace cpu

struct NoClamp {
    static constexpr bool enabled = false;

    // Generic implementation
    template <class T>
    static constexpr T apply(const T v) {
        return v;
    }

    // Specialize for optimization
    template <class T, class R>
    static R apply(const T v);
};

template <class TI, class TO>
struct Clamp {
    static constexpr bool enabled = true;

    // Generic implementation
    static TO apply(const TI v) {
        constexpr auto lo = std::numeric_limits<TO>::lowest();
        constexpr auto hi = std::numeric_limits<TO>::max();

        return (v < lo) ? lo : (v > hi) ? hi : detail::convert<TI, TO>(v);
    }

    // Specialize for optimization
    template <class T, class R>
    static R apply(const T v);
};

template <class TI, class TO>
struct Converter {
    static constexpr size_t vec_f32_size = 32 / sizeof(float);

    // Generic implementation to convert tail elements
    template <class Clamp>
    static void tail(const TI* in, TO* out, size_t n) {
        std::transform(in, in + n, out, [](const TI v) {
            return detail::convert<decltype(Clamp::apply(v)), TO>(Clamp::apply(v));
        });
    }

    // Helper struct to defined optimized version of conversion
    template <class C>
    struct Optimized {
        static constexpr bool enabled = false;
        static void run(const TI* in, TO* out) {}
    };

    // Generic implementation of conversion
    template <class C, typename std::enable_if<!Optimized<C>::enabled>::type* = nullptr>
    static void apply(const TI* in, TO* out, size_t n) {
        return tail<C>(in, out, n);
    }

    // Enabled when Optimized struct specialized for optimizations
    template <class C, typename std::enable_if<Optimized<C>::enabled>::type* = nullptr>
    static void apply(const TI* in, TO* out, size_t n) {
        if (cpu::may_i_use(cpu::isa_t::avx2)) {
            for (; n >= vec_f32_size; n -= vec_f32_size, in += vec_f32_size, out += vec_f32_size) {
                Optimized<C>::run(in, out);
            }
        }
        tail<C>(in, out, n);
    }
};

}  // namespace reference
}  // namespace ov
