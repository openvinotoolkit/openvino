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
template <typename T>
T clamp_pixel(float a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::min(std::max(std::round(a), 0.0f), 255.0f));
    } else {
        return static_cast<T>(std::min(std::max(a, 0.0f), 255.0f));
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

// Reference ITU-R BT.601-7 Section 2.5 for information on how
// the constants in the following two functions were derived.

/// Convert a single YUV pixel (BT.601 limited range) to (R, G, B).
template <typename T, typename U = float>
std::tuple<T, T, T> yuv_pixel_to_rgb(U y_val, U u_val, U v_val) {
    const auto c = static_cast<float>(y_val) - 16.f;
    const auto d = static_cast<float>(u_val) - 128.f;
    const auto e = static_cast<float>(v_val) - 128.f;
    return {clamp_pixel<T>(1.164f * c + 1.596f * e),
            clamp_pixel<T>(1.164f * c - 0.391f * d - 0.813f * e),
            clamp_pixel<T>(1.164f * c + 2.018f * d)};
}

/// Convert a single RGB pixel to (Y, U, V) using BT.601 limited range.
template <typename T, typename U = float>
std::tuple<T, T, T> rgb_pixel_to_yuv(U r_val, U g_val, U b_val) {
    const auto r = static_cast<float>(r_val);
    const auto g = static_cast<float>(g_val);
    const auto b = static_cast<float>(b_val);
    return {clamp_pixel<T>(0.257f * r + 0.504f * g + 0.098f * b + 16.f),
            clamp_pixel<T>(-0.148f * r - 0.291f * g + 0.439f * b + 128.f),
            clamp_pixel<T>(0.439f * r - 0.368f * g - 0.071f * b + 128.f)};
}

}  // namespace ov::reference
