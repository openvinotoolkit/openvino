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
    for (int batch = 0; batch < batch_size; batch++) {
        T* out = out_ptr + batch * image_w * image_h * 3;
        auto y_ptr = arg_y + batch * stride_y;
        auto uv_ptr = arg_uv + batch * stride_uv;
        for (int h = 0; h < image_h; h++) {
            for (int w = 0; w < image_w; w++) {
                auto y_index = h * image_w + w;
                auto y_val = static_cast<float>(y_ptr[y_index]);
                auto uv_index = (h / 2) * image_w + (w / 2) * 2;
                auto u_val = static_cast<float>(uv_ptr[uv_index]);
                auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
                auto c = y_val - 16.f;
                auto d = u_val - 128.f;
                auto e = v_val - 128.f;
                auto clip = [](float a) -> T {
                    if (std::is_integral<T>()) {
                        return static_cast<T>(std::min(std::max(std::round(a), 0.f), 255.f));
                    } else {
                        return static_cast<T>(std::min(std::max(a, 0.f), 255.f));
                    }
                };
                auto b = clip(1.164f * c + 2.018f * d);
                auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
                auto r = clip(1.164f * c + 1.596f * e);
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
