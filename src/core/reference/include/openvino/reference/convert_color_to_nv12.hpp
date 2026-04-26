// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/op/util/convert_color_to_nv12_base.hpp"

namespace ov {
namespace reference {

template <typename T>
std::tuple<T, T, T> rgb_pixel_to_yuv(float r_val, float g_val, float b_val) {
    auto clip = [](float a) -> T {
        if constexpr (std::is_integral<T>::value) {
            return static_cast<T>(std::min(std::max(std::round(a), 0.f), 255.f));
        } else {
            return static_cast<T>(std::min(std::max(a, 0.f), 255.f));
        }
    };
    auto y = clip(0.257f * r_val + 0.504f * g_val + 0.098f * b_val + 16.f);
    auto u = clip(-0.148f * r_val - 0.291f * g_val + 0.439f * b_val + 128.f);
    auto v = clip(0.439f * r_val - 0.368f * g_val - 0.071f * b_val + 128.f);
    return std::tuple<T, T, T>{y, u, v};
}

template <typename T>
void color_convert_to_nv12(const T* rgb_ptr,
                           T* out_y,
                           T* out_uv,
                           size_t batch_size,
                           size_t image_h,
                           size_t image_w,
                           size_t stride_y,
                           size_t stride_uv,
                           ov::op::util::ConvertColorToNV12Base::ColorConversion color_format) {
    const bool is_rgb = (color_format == ov::op::util::ConvertColorToNV12Base::ColorConversion::RGB_TO_NV12);
    const size_t r_offset = is_rgb ? 0 : 2;
    const size_t b_offset = is_rgb ? 2 : 0;

    auto round_cast = [](float a) -> T {
        if constexpr (std::is_integral<T>::value) {
            return static_cast<T>(std::round(a));
        } else {
            return static_cast<T>(a);
        }
    };
    for (size_t batch = 0; batch < batch_size; batch++) {
        const T* rgb = rgb_ptr + batch * image_w * image_h * 3;
        T* y_ptr = out_y + batch * stride_y;
        T* uv_ptr = out_uv + batch * stride_uv;
        for (size_t h = 0; h < image_h; h += 2) {
            for (size_t w = 0; w < image_w; w += 2) {
                float u_sum = 0.f;
                float v_sum = 0.f;
                for (size_t dh = 0; dh < 2; dh++) {
                    for (size_t dw = 0; dw < 2; dw++) {
                        size_t pixel_idx = (h + dh) * image_w + (w + dw);
                        size_t rgb_idx = pixel_idx * 3;
                        float r = static_cast<float>(rgb[rgb_idx + r_offset]);
                        float g = static_cast<float>(rgb[rgb_idx + 1]);
                        float b = static_cast<float>(rgb[rgb_idx + b_offset]);
                        T y, u, v;
                        std::tie(y, u, v) = rgb_pixel_to_yuv<T>(r, g, b);
                        y_ptr[pixel_idx] = y;
                        u_sum += static_cast<float>(u);
                        v_sum += static_cast<float>(v);
                    }
                }
                size_t uv_index = (h / 2) * image_w + w;
                uv_ptr[uv_index] = round_cast(u_sum / 4.f);
                uv_ptr[uv_index + 1] = round_cast(v_sum / 4.f);
            }
        }
    }
}

}  // namespace reference
}  // namespace ov
