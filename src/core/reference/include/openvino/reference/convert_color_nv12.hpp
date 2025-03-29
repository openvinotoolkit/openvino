// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/util/convert_color_i420_base.hpp"
#include "openvino/op/util/convert_color_nv12_base.hpp"

namespace ov {
namespace reference {

template <typename T>
std::tuple<T, T, T> yuv_pixel_to_rgb(float y_val, float u_val, float v_val) {
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
    return std::tuple<T, T, T>{r, g, b};
}

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
    for (size_t batch = 0; batch < batch_size; batch++) {
        T* out = out_ptr + batch * image_w * image_h * 3;
        auto y_ptr = arg_y + batch * stride_y;
        auto uv_ptr = arg_uv + batch * stride_uv;
        for (size_t h = 0; h < image_h; h++) {
            for (size_t w = 0; w < image_w; w++) {
                auto y_index = h * image_w + w;
                auto y_val = static_cast<float>(y_ptr[y_index]);
                auto uv_index = (h / 2) * image_w + (w / 2) * 2;
                auto u_val = static_cast<float>(uv_ptr[uv_index]);
                auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
                T r, g, b;
                std::tie(r, g, b) = yuv_pixel_to_rgb<T>(y_val, u_val, v_val);
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

template <typename T>
void color_convert_i420(const T* arg_y,
                        const T* arg_u,
                        const T* arg_v,
                        T* out_ptr,
                        size_t batch_size,
                        size_t image_h,
                        size_t image_w,
                        size_t stride_y,
                        size_t stride_uv,
                        ov::op::util::ConvertColorI420Base::ColorConversion color_format) {
    for (size_t batch = 0; batch < batch_size; batch++) {
        T* out = out_ptr + batch * image_w * image_h * 3;
        auto y_ptr = arg_y + batch * stride_y;
        auto u_ptr = arg_u + batch * stride_uv;
        auto v_ptr = arg_v + batch * stride_uv;
        for (size_t h = 0; h < image_h; h++) {
            for (size_t w = 0; w < image_w; w++) {
                auto y_index = h * image_w + w;
                auto y_val = static_cast<float>(y_ptr[y_index]);
                auto uv_index = (h / 2) * (image_w / 2) + (w / 2);
                auto u_val = static_cast<float>(u_ptr[uv_index]);
                auto v_val = static_cast<float>(v_ptr[uv_index]);
                T r, g, b;
                std::tie(r, g, b) = yuv_pixel_to_rgb<T>(y_val, u_val, v_val);
                if (color_format == ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB) {
                    out[y_index * 3] = r;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = b;
                } else if (color_format == ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR) {
                    out[y_index * 3] = b;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = r;
                }
            }
        }
    }
}

template <ov::element::Type_t T>
inline bool color_convert_nv12(const std::shared_ptr<Node>& op,
                               ov::TensorVector& outputs,
                               const ov::TensorVector& inputs,
                               ov::op::util::ConvertColorNV12Base::ColorConversion type) {
    using ET = typename ov::element_type_traits<T>::value_type;
    static const size_t N_DIM = 0;
    static const size_t H_DIM = 1;
    static const size_t W_DIM = 2;
    OPENVINO_ASSERT(op->get_input_size() == 1 || op->get_input_size() == 2,
                    "NV12 conversion shall have one or 2 inputs, but it is ",
                    op->get_input_size());
    auto single_plane = op->get_input_size() == 1;

    const auto& y_tensor = inputs[0];
    auto batch_size = y_tensor.get_shape()[N_DIM];
    auto image_w = y_tensor.get_shape()[W_DIM];
    auto image_h = y_tensor.get_shape()[H_DIM];
    if (single_plane) {
        image_h = image_h * 2 / 3;
    }
    outputs[0].set_shape({batch_size, image_h, image_w, 3});  // 3 is RGB
    if (single_plane) {
        color_convert_nv12(y_tensor.data<ET>(),
                           y_tensor.data<ET>() + image_w * image_h,
                           outputs[0].data<ET>(),
                           batch_size,
                           image_h,
                           image_w,
                           image_w * image_h * 3 / 2,
                           image_w * image_h * 3 / 2,
                           type);
    } else {
        const auto& uv_tensor = inputs[1];
        color_convert_nv12(y_tensor.data<ET>(),
                           uv_tensor.data<ET>(),
                           outputs[0].data<ET>(),
                           batch_size,
                           image_h,
                           image_w,
                           image_w * image_h,
                           image_w * image_h / 2,
                           type);
    }
    return true;
}

template <ov::element::Type_t T>
inline bool color_convert_i420(const std::shared_ptr<Node>& op,
                               ov::TensorVector& outputs,
                               const ov::TensorVector& inputs,
                               ov::op::util::ConvertColorI420Base::ColorConversion type) {
    using ET = typename ov::element_type_traits<T>::value_type;
    static const size_t N_DIM = 0;
    static const size_t H_DIM = 1;
    static const size_t W_DIM = 2;
    OPENVINO_ASSERT(op->get_input_size() == 1 || op->get_input_size() == 3,
                    "I420 conversion shall have one or 3 inputs, but it is ",
                    op->get_input_size());
    auto single_plane = op->get_input_size() == 1;

    const auto& y_tensor = inputs[0];
    auto batch_size = y_tensor.get_shape()[N_DIM];
    auto image_w = y_tensor.get_shape()[W_DIM];
    auto image_h = y_tensor.get_shape()[H_DIM];
    if (single_plane) {
        image_h = image_h * 2 / 3;
    }
    outputs[0].set_shape({batch_size, image_h, image_w, 3});  // 3 is RGB
    if (single_plane) {
        color_convert_i420(y_tensor.data<ET>(),
                           y_tensor.data<ET>() + image_w * image_h,
                           y_tensor.data<ET>() + 5 * image_w * image_h / 4,
                           outputs[0].data<ET>(),
                           batch_size,
                           image_h,
                           image_w,
                           image_w * image_h * 3 / 2,
                           image_w * image_h * 3 / 2,
                           type);
    } else {
        const auto& u_tensor = inputs[1];
        const auto& v_tensor = inputs[2];
        color_convert_i420(y_tensor.data<ET>(),
                           u_tensor.data<ET>(),
                           v_tensor.data<ET>(),
                           outputs[0].data<ET>(),
                           batch_size,
                           image_h,
                           image_w,
                           image_w * image_h,
                           image_w * image_h / 4,
                           type);
    }
    return true;
}

}  // namespace reference
}  // namespace ov
