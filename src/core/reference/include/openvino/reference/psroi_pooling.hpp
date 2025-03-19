// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <string>

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
enum PSROIPoolingMode { AVG, BILINEAR };
template <typename T>
void psroi_pooling(const T* input,
                   const Shape& input_shape,
                   const T* rois,
                   const Shape& rois_shape,
                   T* output,
                   const Shape& output_shape,
                   const std::string& mode_str,
                   float spatial_scale,
                   int spatial_bins_x,
                   int spatial_bins_y) {
    PSROIPoolingMode mode;
    if (mode_str == "average") {
        mode = AVG;
    } else if (mode_str == "bilinear") {
        mode = BILINEAR;
    } else {
        OPENVINO_ASSERT(false, "Invalid PS ROI pooling mode: " + mode_str);
    }
    size_t channels_in = input_shape[1];
    size_t height = input_shape[2];
    size_t width = input_shape[3];
    size_t num_rois = output_shape[0];
    size_t channels_out = output_shape[1];
    size_t pooling_height = output_shape[2];
    size_t pooling_width = output_shape[3];
    int num_spatial_bins = spatial_bins_x * spatial_bins_y;
    for (size_t roi = 0; roi < num_rois; roi++) {
        const T* box = rois + roi * 5;
        int batch_id = static_cast<int>(box[0]);
        float start_w = 0;
        float start_h = 0;
        float end_w = 0;
        float end_h = 0;
        if (mode == BILINEAR) {
            start_w = static_cast<float>(box[1]) * spatial_scale;
            start_h = static_cast<float>(box[2]) * spatial_scale;
            end_w = static_cast<float>(box[3]) * spatial_scale;
            end_h = static_cast<float>(box[4]) * spatial_scale;
        } else if (mode == AVG) {
            start_w = std::roundf(static_cast<float>(box[1])) * spatial_scale;
            start_h = std::roundf(static_cast<float>(box[2])) * spatial_scale;
            end_w = (std::roundf(static_cast<float>(box[3])) + 1.0f) * spatial_scale;
            end_h = (std::roundf(static_cast<float>(box[4])) + 1.0f) * spatial_scale;
        }
        float box_width = end_w - start_w;
        float box_height = end_h - start_h;
        float bin_width = box_width / pooling_width;
        float bin_height = box_height / pooling_height;
        float width_scale = 0;
        float height_scale = 0;
        if (mode == BILINEAR) {
            bin_width = box_width / spatial_bins_x;
            bin_height = box_height / spatial_bins_y;
            if (pooling_width > 1)
                width_scale = bin_width * (width - 1) / (pooling_width - 1);
            if (pooling_height > 1)
                height_scale = bin_height * (height - 1) / (pooling_height - 1);
        }
        size_t c_in = 0;
        for (size_t c_out = 0; c_out < channels_out; c_out++) {
            for (size_t ph = 0; ph < pooling_height; ph++) {
                for (size_t pw = 0; pw < pooling_width; pw++) {
                    size_t index = ((roi * channels_out + c_out) * pooling_height + ph) * pooling_width + pw;
                    output[index] = 0;
                    if (mode == AVG) {
                        size_t bin_start_w = std::min(static_cast<size_t>(floorf(start_w + pw * bin_width)), width - 1);
                        size_t bin_start_h =
                            std::min(static_cast<size_t>(floorf(start_h + ph * bin_height)), height - 1);
                        size_t current_bin_width =
                            std::min(static_cast<size_t>(ceilf(start_w + (pw + 1) * bin_width)), width) - bin_start_w;
                        size_t current_bin_height =
                            std::min(static_cast<size_t>(ceilf(start_h + (ph + 1) * bin_height)), height) - bin_start_h;
                        T sum = 0;
                        const T* input_offset =
                            input + ((batch_id * channels_in + c_in) * height + bin_start_h) * width + bin_start_w;
                        for (size_t h = 0; h < current_bin_height; h++) {
                            for (size_t w = 0; w < current_bin_width; w++) {
                                sum += input_offset[h * width + w];
                            }
                        }
                        output[index] = sum / static_cast<T>(current_bin_width * current_bin_height);
                        c_in++;
                    } else if (mode == BILINEAR) {
                        c_in = 0;
                        for (int sby = 0; sby < spatial_bins_y; sby++) {
                            for (int sbx = 0; sbx < spatial_bins_x; sbx++) {
                                float bin_start_w = start_w + sbx * bin_width;
                                float bin_start_h = start_h + sby * bin_height;

                                const T* input_offset =
                                    input + (batch_id * channels_in + c_in * channels_out + c_out) * height * width;
                                float point_x = pooling_width > 1
                                                    ? (pw * width_scale + bin_start_w * (width - 1))
                                                    : (bin_start_w + bin_start_w + bin_width) * (width - 1) / 2;
                                float point_y = pooling_height > 1
                                                    ? (ph * height_scale + bin_start_h * (height - 1))
                                                    : (bin_start_h + bin_start_h + bin_height) * (height - 1) / 2;
                                if (point_x < width && point_y < height) {
                                    size_t left = static_cast<size_t>(floorf(point_x));
                                    size_t right = std::min(static_cast<size_t>(ceilf(point_x)), width - 1);
                                    size_t top = static_cast<size_t>(floorf(point_y));
                                    size_t bottom = std::min(static_cast<size_t>(ceilf(point_y)), height - 1);
                                    T top_left = input_offset[top * width + left];
                                    T top_right = input_offset[top * width + right];
                                    T bottom_left = input_offset[bottom * width + left];
                                    T bottom_right = input_offset[bottom * width + right];

                                    T top_interp = top_left + (top_right - top_left) * static_cast<T>(point_x - left);
                                    T bottom_interp =
                                        bottom_left + (bottom_right - bottom_left) * static_cast<T>(point_x - left);
                                    output[index] +=
                                        top_interp + (bottom_interp - top_interp) * static_cast<T>(point_y - top);
                                }
                                c_in++;
                            }
                        }
                        output[index] /= num_spatial_bins;
                    }
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
