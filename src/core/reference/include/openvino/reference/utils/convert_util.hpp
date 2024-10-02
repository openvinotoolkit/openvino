// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <type_traits>

#include "openvino/reference/convert.hpp"

namespace ov {

// forward declare from inference dev API (cannot be included)
extern bool with_cpu_x86_avx2();

namespace reference {

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
    static constexpr TO apply(const TI v) {
        return (v < std::numeric_limits<TO>::lowest())
                   ? std::numeric_limits<TO>::lowest()
                   : ((v > std::numeric_limits<TO>::max()) ? std::numeric_limits<TO>::max()
                                                           : detail::convert<TI, TO>(v));
    }

    // Specialize for optimization
    template <class T, class R>
    static R apply(const T v);
};

template <class TI, class TO>
struct Converter {
    static constexpr size_t vec_f32_size = 32 / sizeof(float);

    // Generic implementation to convert tail elements
    template <class ClampMode>
    static void tail(const TI* in, TO* out, size_t n) {
        std::transform(in, in + n, out, [](const TI v) {
            return detail::convert<decltype(ClampMode::apply(v)), TO>(ClampMode::apply(v));
        });
    }

    // Helper struct to defined optimized version of conversion
    template <class ClampMode>
    struct Optimized {
        static constexpr bool enabled = false;
        static void run(const TI* in, TO* out) {}
    };

    // Generic implementation of conversion
    template <class ClampMode, typename std::enable_if<!Optimized<ClampMode>::enabled>::type* = nullptr>
    static void apply(const TI* in, TO* out, size_t n) {
        return tail<ClampMode>(in, out, n);
    }

    // Enabled when Optimized struct specialized defined for optimization
    template <class ClampMode, typename std::enable_if<Optimized<ClampMode>::enabled>::type* = nullptr>
    static void apply(const TI* in, TO* out, size_t n) {
        if (with_cpu_x86_avx2()) {
            for (; n >= vec_f32_size; n -= vec_f32_size, in += vec_f32_size, out += vec_f32_size) {
                Optimized<ClampMode>::run(in, out);
            }
        }
        tail<ClampMode>(in, out, n);
    }
};

}  // namespace reference
}  // namespace ov
