// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/op/util/convert_color_nv12_base.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void color_convert_nv12(const T* arg_y,
                        const T* arg_uv,
                        T* out_ptr,
                        size_t batch_size,
                        size_t image_h,
                        size_t image_w,
                        size_t stride_y,
                        size_t stride_uv,
                        ov::op::util::ConvertColorNV12Base::ColorConversion color_format) {
    // With C++20 - std::endian can be used at compile time
    auto little_endian = []() -> int {
        union {
            int32_t i;
            char c[4];
        } u = {0x00000001};
        return static_cast<int>(u.c[0]);
    };
    std::cout << "color_convert_nv12_dbg start " << little_endian << std::endl;
    std::cout << "Y=" << static_cast<const void*>(arg_y) << " UV=" << static_cast<const void*>(arg_uv)
              << " out=" << static_cast<const void*>(out_ptr) << std::endl;
    std::cout << "N=" << batch_size << " h=" << image_h << " w=" << image_w << " sY = " << stride_y
              << " sUV=" << stride_uv << std::endl;
    auto is_little_endian = little_endian();
    for (int batch = 0; batch < batch_size; batch++) {
        T* out = out_ptr + batch * image_w * image_h;
        auto y_ptr = arg_y + batch * stride_y;
        auto uv_ptr = arg_uv + batch * stride_uv;
        for (int h = 0; h < image_h; h++) {
            for (int w = 0; w < image_w; w++) {
                auto y_index = h * image_w + w;
                // For little-endian systems:
                //      Y bytes are shuffled as Y1, Y0, Y3, Y2, Y5, Y4, etc.
                //      UV bytes are ordered as V0, U0, V1, U1, V2, U2, etc.
                // For float point case follow the same order
                auto add_y_index = is_little_endian ? (w % 2 ? -1 : 1) : 0;
                auto y_val = static_cast<float>(y_ptr[y_index + add_y_index]);
                auto uv_index = (h / 2) * image_w + (w / 2) * 2;
                auto u_val = static_cast<float>(uv_ptr[uv_index + is_little_endian]);
                auto v_val = static_cast<float>(uv_ptr[uv_index + 1 - is_little_endian]);
                auto c = y_val - 16.f;
                auto d = u_val - 128.f;
                auto e = v_val - 128.f;
                auto clip = [](float a) -> T {
                    return a < 0.5f ? static_cast<T>(0) : (a > 254.5f ? static_cast<T>(255) : static_cast<T>(a));
                };
                auto b = clip(1.164f * c + 2.018f * d);
                auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
                auto r = clip(1.164f * c + 1.596f * e);
                std::cout << y_index << "[" << static_cast<float>(y_val) << " " << static_cast<float>(u_val) << " "
                          << static_cast<float>(v_val) << "][ " << static_cast<float>(r) << " " << static_cast<float>(g)
                          << " " << static_cast<float>(b) << "] " << std::endl;
                std::flush(std::cout);
                if (color_format == ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_RGB) {
                    out[y_index * 3] = r;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = b;
                } else if (color_format == ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR) {
                    out[y_index * 3] = b;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = r;
                }
            }
        }
    }
}

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
