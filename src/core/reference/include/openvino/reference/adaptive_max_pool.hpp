// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/adaptive_avg_pool.hpp"

namespace ov {
namespace reference {
template <typename T, typename IT>
void adaptive_max_pool_1d(const T* arg, T* out, IT* indices, size_t h_in, size_t h_out) {
    for (size_t i = 0; i < h_out; i++) {
        auto from = arg + adaptive_pool::window_start(i, h_in, h_out);
        auto to = arg + adaptive_pool::window_end(i, h_in, h_out);
        OPENVINO_ASSERT(to - from != 0, "AdaptiveMaxPool elements == 0, must be non-zero");
        auto it = std::max_element(from, to);
        out[i] = static_cast<T>(*it);
        indices[i] = static_cast<IT>(it - arg);
    }
}
template <typename T, typename IT>
void adaptive_max_pool_2d(const T* arg, T* out, IT* indices, size_t h_in, size_t h_out, size_t w_in, size_t w_out) {
    for (size_t i = 0; i < h_out; i++) {
        size_t h_start = adaptive_pool::window_start(i, h_in, h_out);
        size_t h_end = adaptive_pool::window_end(i, h_in, h_out);
        for (size_t j = 0; j < w_out; j++) {
            size_t w_start = adaptive_pool::window_start(j, w_in, w_out);
            size_t w_end = adaptive_pool::window_end(j, w_in, w_out);
            OPENVINO_ASSERT((w_end - w_start) * (h_end - h_start) != 0,
                            "AdaptiveMaxPool elements == 0, must be non-zero");
            auto result = arg + h_start * w_in + w_start;
            for (size_t n = h_start; n < h_end; n++) {
                auto from = arg + n * w_in + w_start;
                auto to = arg + n * w_in + w_end;
                auto it = std::max_element(from, to);
                result = *it > *result ? it : result;
            }
            out[i * w_out + j] = static_cast<T>(*result);
            indices[i * w_out + j] = static_cast<IT>(result - arg);
        }
    }
}
template <typename T, typename IT>
void adaptive_max_pool_3d(const T* arg,
                          T* out,
                          IT* indices,
                          size_t d_in,
                          size_t d_out,
                          size_t h_in,
                          size_t h_out,
                          size_t w_in,
                          size_t w_out) {
    for (size_t i = 0; i < d_out; i++) {
        size_t d_start = adaptive_pool::window_start(i, d_in, d_out);
        size_t d_end = adaptive_pool::window_end(i, d_in, d_out);
        for (size_t j = 0; j < h_out; j++) {
            size_t h_start = adaptive_pool::window_start(j, h_in, h_out);
            size_t h_end = adaptive_pool::window_end(j, h_in, h_out);
            for (size_t k = 0; k < w_out; k++) {
                size_t w_start = adaptive_pool::window_start(k, w_in, w_out);
                size_t w_end = adaptive_pool::window_end(k, w_in, w_out);
                OPENVINO_ASSERT((w_end - w_start) * (h_end - h_start) != 0,
                                "AdaptiveMaxPool elements == 0, must be non-zero");
                auto result = arg + d_start * h_in * w_in + h_start * w_in + w_start;
                for (size_t n = d_start; n < d_end; n++) {
                    for (size_t m = h_start; m < h_end; m++) {
                        auto from = arg + n * h_in * w_in + m * w_in + w_start;
                        auto to = arg + n * h_in * w_in + m * w_in + w_end;
                        auto it = std::max_element(from, to);
                        result = *it > *result ? it : result;
                    }
                }
                out[i * h_out * w_out + j * w_out + k] = static_cast<T>(*result);
                indices[i * h_out * w_out + j * w_out + k] = static_cast<IT>(result - arg);
            }
        }
    }
}
template <typename T, typename IT>
void adaptive_max_pool(const T* arg, T* out, IT* selected_indices, const Shape& arg_shape, const Shape& out_shape) {
    OPENVINO_ASSERT(arg_shape.size() == out_shape.size() && 2 < arg_shape.size() && arg_shape.size() < 6,
                    "AdaptiveAvgPool supports only 3D, 4D and 5D input shape");
    size_t channel_size = 1;
    for (size_t i = 2; i < arg_shape.size(); i++) {
        channel_size *= arg_shape[i];
    }
    size_t batch_size = arg_shape[1] * channel_size;
    size_t out_channel_size = 1;
    for (size_t i = 2; i < out_shape.size(); i++) {
        out_channel_size *= out_shape[i];
    }
    size_t out_batch_size = arg_shape[1] * out_channel_size;
    for (size_t b = 0; b < arg_shape[0]; b++) {
        for (size_t c = 0; c < arg_shape[1]; c++) {
            auto arg_pos = arg + b * batch_size + c * channel_size;
            auto out_pos = out + b * out_batch_size + c * out_channel_size;
            auto sel_ind_pos = selected_indices + b * out_batch_size + c * out_channel_size;
            if (arg_shape.size() == 3) {
                adaptive_max_pool_1d<T>(arg_pos, out_pos, sel_ind_pos, arg_shape[2], out_shape[2]);
            } else if (arg_shape.size() == 4) {
                adaptive_max_pool_2d<T>(arg_pos,
                                        out_pos,
                                        sel_ind_pos,
                                        arg_shape[2],
                                        out_shape[2],
                                        arg_shape[3],
                                        out_shape[3]);
            } else if (arg_shape.size() == 5) {
                adaptive_max_pool_3d<T>(arg_pos,
                                        out_pos,
                                        sel_ind_pos,
                                        arg_shape[2],
                                        out_shape[2],
                                        arg_shape[3],
                                        out_shape[3],
                                        arg_shape[4],
                                        out_shape[4]);
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
