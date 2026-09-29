// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

namespace ov::reference {

/// Clamp \p a to [0, 255]. For integral types the value is rounded first.
template <typename T, typename U = float>
T clip(U a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::min(std::max(std::round(a), U{0}), U{255}));
    } else {
        return static_cast<T>(std::min(std::max(a, U{0}), U{255}));
    }
}

/// Cast \p a to T, rounding first for integral types.
template <typename T, typename U>
T round_cast(U a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::round(a));
    } else {
        return static_cast<T>(a);
    }
}

/// Convert a single YUV pixel (BT.601 limited range) to (R, G, B).
template <typename T, typename U = float>
std::tuple<T, T, T> yuv_pixel_to_rgb(U y_val, U u_val, U v_val) {
    const auto c = y_val - 16.f;
    const auto d = u_val - 128.f;
    const auto e = v_val - 128.f;
    return {clip<T>(1.164f * c + 1.596f * e),
            clip<T>(1.164f * c - 0.391f * d - 0.813f * e),
            clip<T>(1.164f * c + 2.018f * d)};
}

/// Convert a single RGB pixel to (Y, U, V).
template <typename T, typename U = float>
std::tuple<T, T, T> rgb_pixel_to_yuv(U r_val, U g_val, U b_val) {
    return {clip<T>(0.257f * r_val + 0.504f * g_val + 0.098f * b_val + 16.f),
            clip<T>(-0.148f * r_val - 0.291f * g_val + 0.439f * b_val + 128.f),
            clip<T>(0.439f * r_val - 0.368f * g_val - 0.071f * b_val + 128.f)};
}

}  // namespace ov::reference
