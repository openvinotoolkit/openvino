// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <tuple>

namespace ov::reference {

namespace func {

template <typename T>
T clip(double a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::min(std::max(std::round(a), 0.0), 255.0));
    } else {
        return static_cast<T>(std::min(std::max(a, 0.0), 255.0));
    }
}

template <typename T>
T round_cast(double a) {
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(std::round(a));
    } else {
        return static_cast<T>(a);
    }
}

template <typename T>
std::tuple<T, T, T> rgb_pixel_to_yuv(T r_val, T g_val, T b_val) {
    const double r = static_cast<double>(r_val);
    const double g = static_cast<double>(g_val);
    const double b = static_cast<double>(b_val);
    return {clip<T>(0.257 * r + 0.504 * g + 0.098 * b + 16.0),
            clip<T>(-0.148 * r - 0.291 * g + 0.439 * b + 128.0),
            clip<T>(0.439 * r - 0.368 * g - 0.071 * b + 128.0)};
}

}  // namespace func

template <typename T, bool IsRGB>
void color_convert_to_nv12(const T* rgb_ptr,
                           T* out_y,
                           T* out_uv,
                           size_t batch_size,
                           size_t image_h,
                           size_t image_w,
                           bool single_plane) {
    constexpr size_t r_offset = IsRGB ? 0 : 2;
    constexpr size_t b_offset = IsRGB ? 2 : 0;

    const size_t frame_size = image_w * image_h;
    const size_t stride_y = single_plane ? frame_size * 3 / 2 : frame_size;
    const size_t stride_uv = single_plane ? frame_size * 3 / 2 : frame_size / 2;
    T* uv_base = single_plane ? out_y + frame_size : out_uv;
    for (size_t batch = 0; batch < batch_size; batch++) {
        const T* rgb = rgb_ptr + batch * frame_size * 3;
        T* y_ptr = out_y + batch * stride_y;
        T* uv_ptr = uv_base + batch * stride_uv;
        for (size_t h = 0; h < image_h; h += 2) {
            for (size_t w = 0; w < image_w; w += 2) {
                double u_sum = 0.0;
                double v_sum = 0.0;
                for (size_t dh = 0; dh < 2; dh++) {
                    for (size_t dw = 0; dw < 2; dw++) {
                        size_t pixel_idx = (h + dh) * image_w + (w + dw);
                        size_t rgb_idx = pixel_idx * 3;
                        T y, u, v;
                        std::tie(y, u, v) = func::rgb_pixel_to_yuv<T>(
                            rgb[rgb_idx + r_offset],
                            rgb[rgb_idx + 1],
                            rgb[rgb_idx + b_offset]);
                        y_ptr[pixel_idx] = y;
                        u_sum += static_cast<double>(u);
                        v_sum += static_cast<double>(v);
                    }
                }
                size_t uv_index = (h / 2) * image_w + w;
                uv_ptr[uv_index] = func::round_cast<T>(u_sum / 4.0);
                uv_ptr[uv_index + 1] = func::round_cast<T>(v_sum / 4.0);
            }
        }
    }
}

}  // namespace ov::reference
