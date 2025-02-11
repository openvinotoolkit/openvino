// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// DeformablePSROIPooling implementation was inspired by
// https://github.com/msracver/Deformable-ConvNets
// Copyright (c) 2017 Microsoft
// SPDX-License-Identifier: MIT

#pragma once

#include <cfenv>
#include <cmath>
#include <string>
#include <vector>

#include "clamp.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T>
void deformable_psroi_pooling(const T* data_input,
                              const Shape& data_input_shape,
                              const T* rois_input,
                              const Shape& rois_input_shape,
                              const T* offsets_input,
                              const Shape& offsets_input_shape,
                              T* output,
                              const Shape& output_shape,
                              const std::string& mode_str,
                              const float spatial_scale,
                              const int64_t spatial_bins_x,
                              const int64_t spatial_bins_y,
                              const float trans_std,
                              const int64_t part_size) {
    const size_t channels_in = data_input_shape[1];
    const size_t height_in = data_input_shape[2];
    const size_t width_in = data_input_shape[3];

    const size_t rois_count = output_shape[0];
    const size_t channels_out = output_shape[1];
    const size_t height_out = output_shape[2];
    const size_t width_out = output_shape[3];

    std::fill(output, output + shape_size(output_shape), T{0});

    // Single ROI is described by (batch_id, x1, y1, x2, y2)
    const size_t roi_attrs_count = 5;

    for (size_t roi_idx = 0; roi_idx < rois_count; ++roi_idx) {
        // Pointer to the beginning of the ROI coords tuple
        const T* roi = rois_input + roi_idx * roi_attrs_count;

        // Index of the corresponding input batch
        int64_t roi_batch_id = static_cast<int64_t>(roi[0]);
        if (roi_batch_id < 0)
            continue;

        // Left top ROI corner
        const float roi_x1 = static_cast<float>(std::round(roi[1])) * spatial_scale - 0.5f;
        const float roi_y1 = static_cast<float>(std::round(roi[2])) * spatial_scale - 0.5f;
        // Right down ROI corner
        const float roi_x2 = static_cast<float>(std::round(roi[3]) + 1.0f) * spatial_scale - 0.5f;
        const float roi_y2 = static_cast<float>(std::round(roi[4]) + 1.0f) * spatial_scale - 0.5f;

        const float roi_width = std::max<float>(roi_x2 - roi_x1, 0.1f);
        const float roi_height = std::max<float>(roi_y2 - roi_y1, 0.1f);

        const float bin_width = roi_width / static_cast<float>(width_out);
        const float bin_height = roi_height / static_cast<float>(height_out);

        size_t c_idx_in = 0;
        for (size_t c_idx_out = 0; c_idx_out < channels_out; ++c_idx_out) {
            for (size_t h_idx_out = 0; h_idx_out < height_out; ++h_idx_out) {
                // Next bin is taken from the next input channel
                for (size_t w_idx_out = 0; w_idx_out < width_out; ++w_idx_out, ++c_idx_in) {
                    const size_t out_value_idx =
                        ((roi_idx * channels_out + c_idx_out) * height_out + h_idx_out) * width_out + w_idx_out;

                    // Left top corner of bin
                    float bin_x1_idx = roi_x1 + w_idx_out * bin_width;
                    float bin_y1_idx = roi_y1 + h_idx_out * bin_height;

                    // Take offsets from optional input
                    if (offsets_input != nullptr && offsets_input_shape.size() == 4) {
                        const auto num_coords = 2;  // (x, y)
                        const size_t coords_sub_channels = offsets_input_shape[1] / num_coords;
                        const size_t class_sub_channels = channels_out / coords_sub_channels;
                        const size_t roi_channel_idx = c_idx_out / class_sub_channels;

                        const size_t off_bin_w_idx = w_idx_out * part_size / width_out;
                        const size_t off_bin_h_idx = h_idx_out * part_size / height_out;

                        const size_t offsets_channel_idx =
                            (roi_idx * coords_sub_channels + roi_channel_idx) * num_coords;

                        const size_t x_offset_idx =
                            (offsets_channel_idx * part_size + off_bin_h_idx) * part_size + off_bin_w_idx;

                        const size_t y_offset_idx =
                            ((offsets_channel_idx + 1) * part_size + off_bin_h_idx) * part_size + off_bin_w_idx;

                        T x_offset_value = offsets_input[x_offset_idx];
                        T y_offset_value = offsets_input[y_offset_idx];

                        x_offset_value *= static_cast<T>(trans_std);
                        y_offset_value *= static_cast<T>(trans_std);

                        // Move bin position by normalized offset values
                        bin_x1_idx += static_cast<float>(x_offset_value) * roi_width;
                        bin_y1_idx += static_cast<float>(y_offset_value) * roi_height;
                    }

                    // Each bin is divided into sub-bins
                    // Values of sub-bins are calculated by bilinear interpolation
                    // Value of single bin is average of its sub-bins
                    const float sub_bin_width = bin_width / static_cast<float>(spatial_bins_x);
                    const float sub_bin_height = bin_height / static_cast<float>(spatial_bins_y);

                    T sub_bins_val_sum = 0;
                    size_t legit_sub_bin_count = 0;
                    for (int sub_bin_h_idx = 0; sub_bin_h_idx < spatial_bins_y; ++sub_bin_h_idx) {
                        float sub_bin_y1_idx = bin_y1_idx + sub_bin_h_idx * sub_bin_height;
                        if (sub_bin_y1_idx < -0.5 || sub_bin_y1_idx > height_in - 0.5)
                            continue;

                        for (int sub_bin_w_idx = 0; sub_bin_w_idx < spatial_bins_x; ++sub_bin_w_idx) {
                            float sub_bin_x1_idx = bin_x1_idx + sub_bin_w_idx * sub_bin_width;
                            if (sub_bin_x1_idx < -0.5 || sub_bin_x1_idx > width_in - 0.5)
                                continue;

                            clamp(&sub_bin_x1_idx, &sub_bin_x1_idx, 0.f, width_in - 1.f, 1);
                            clamp(&sub_bin_y1_idx, &sub_bin_y1_idx, 0.f, height_in - 1.f, 1);

                            // Calculate value for sub-bin by bilinear interpolation
                            const int64_t left_x = static_cast<int64_t>(std::floor(sub_bin_x1_idx));
                            const int64_t right_x = static_cast<int64_t>(std::ceil(sub_bin_x1_idx));
                            const int64_t top_y = static_cast<int64_t>(std::floor(sub_bin_y1_idx));
                            const int64_t bottom_y = static_cast<int64_t>(std::ceil(sub_bin_y1_idx));

                            const T* data_channel_ptr =
                                data_input + (roi_batch_id * channels_in + c_idx_in) * height_in * width_in;

                            const T top_left_sample = data_channel_ptr[top_y * width_in + left_x];
                            const T top_right_sample = data_channel_ptr[top_y * width_in + right_x];
                            const T bottom_left_sample = data_channel_ptr[bottom_y * width_in + left_x];
                            const T bottom_right_sample = data_channel_ptr[bottom_y * width_in + right_x];

                            const float delta_left_x = std::fabs(sub_bin_x1_idx - left_x);
                            const float delta_top_y = std::fabs(sub_bin_y1_idx - top_y);

                            const T top_interp =
                                top_left_sample + (top_right_sample - top_left_sample) * static_cast<T>(delta_left_x);
                            const T bottom_interp = bottom_left_sample + (bottom_right_sample - bottom_left_sample) *
                                                                             static_cast<T>(delta_left_x);

                            const T sub_bin_value =
                                top_interp + (bottom_interp - top_interp) * static_cast<T>(delta_top_y);

                            legit_sub_bin_count++;
                            sub_bins_val_sum += sub_bin_value;
                        }
                    }
                    // Calculate average of sub_bin values for single ROI bin
                    if (legit_sub_bin_count != 0) {
                        output[out_value_idx] = sub_bins_val_sum / static_cast<T>(legit_sub_bin_count);
                    }
                }
            }
        }
    }
}
}  // namespace reference
}  // namespace ov
