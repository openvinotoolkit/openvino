//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void roi_pooling(const T* feature_maps,
                             const T* rois,
                             T* output,
                             const Shape& feature_maps_shape,
                             const Shape& rois_shape,
                             const Shape& output_shape,
                             const float spatial_scale,
                             const std::string& pooling_method)
            {
                // Feature maps input shape: {N, C, H, W}
                const int batches = feature_maps_shape[0];
                const int channels = feature_maps_shape[1];
                const int height = feature_maps_shape[2];
                const int width = feature_maps_shape[3];

                // Output shape: {NUM_ROIS, C, pooled_h, pooled_w}
                const int pooled_h = output_shape[2];
                const int pooled_w = output_shape[3];

                // ROIs shape: {NUM_ROIS, 5}
                const int num_rois = rois_shape[0];

                for (unsigned int roi_num = 0; roi_num < num_rois; roi_num++)
                {
                    // ROI tuple: [roi_batch_id, roi_w_start, roi_h_start, roi_w_end, roi_h_end]
                    // ROI index
                    int roi_idx = rois_shape[1] * roi_num;

                    // ROI batch id
                    int roi_batch_id = rois[roi_idx + 0];

                    // ROI batch id must be in the range of [0, N-1]
                    NGRAPH_CHECK(0 <= roi_batch_id && roi_batch_id < batches,
                                 "ROI batch id must be in the range of [0, N-1]");

                    if (pooling_method == "max")
                    {
                        // ROI coordinates scaled to input feature maps
                        int roi_w_start = std::round(rois[roi_idx + 1] * spatial_scale);
                        int roi_h_start = std::round(rois[roi_idx + 2] * spatial_scale);
                        int roi_w_end = std::round(rois[roi_idx + 3] * spatial_scale);
                        int roi_h_end = std::round(rois[roi_idx + 4] * spatial_scale);

                        // Force malformed ROIs to be 1x1
                        int roi_height = std::max(roi_h_end - roi_h_start + 1, 1);
                        int roi_width = std::max(roi_w_end - roi_w_start + 1, 1);

                        // Divide ROIs into sub-regions for max pooling
                        T bin_size_h = static_cast<T>(roi_height) / pooled_h;
                        T bin_size_w = static_cast<T>(roi_width) / pooled_w;

                        const T* batch_data =
                            feature_maps + roi_batch_id * channels * height * width;

                        for (unsigned int c = 0; c < channels; c++)
                        {
                            for (unsigned int ph = 0; ph < pooled_h; ph++)
                            {
                                for (unsigned int pw = 0; pw < pooled_w; pw++)
                                {
                                    // Compute pooling region for this output unit:
                                    //  start (included) = floor(ph * roi_height / pooled_h)
                                    //  end (excluded) = ceil((ph + 1) * roi_height / pooled_h)
                                    int h_start = static_cast<int>(
                                        std::floor(static_cast<T>(ph) * bin_size_h));
                                    int w_start = static_cast<int>(
                                        std::floor(static_cast<T>(pw) * bin_size_w));
                                    int h_end = static_cast<int>(
                                        std::ceil(static_cast<T>(ph + 1) * bin_size_h));
                                    int w_end = static_cast<int>(
                                        std::ceil(static_cast<T>(pw + 1) * bin_size_w));

                                    // Add ROI offsets and clip to input boundaries
                                    h_start = std::min(std::max(h_start + roi_h_start, 0), height);
                                    w_start = std::min(std::max(w_start + roi_w_start, 0), width);
                                    h_end = std::min(std::max(h_end + roi_h_start, 0), height);
                                    w_end = std::min(std::max(w_end + roi_w_start, 0), width);

                                    const size_t pool_index =
                                        roi_num * channels * pooled_h * pooled_w +
                                        c * pooled_h * pooled_w + ph * pooled_w + pw;

                                    // Define an empty pooling region to be zero
                                    bool is_empty = (h_end <= h_start) || (w_end <= w_start);
                                    output[pool_index] = is_empty
                                                             ? static_cast<T>(0)
                                                             : std::numeric_limits<T>::lowest();

                                    for (unsigned int h = h_start; h < h_end; h++)
                                    {
                                        for (unsigned int w = w_start; w < w_end; w++)
                                        {
                                            const size_t index = h * width + w;
                                            output[pool_index] =
                                                std::max(batch_data[index], output[pool_index]);
                                        }
                                    }
                                }
                            }
                            // Increment batch data pointer by one channel
                            batch_data += height * width;
                        }
                    }
                    else if (pooling_method == "bilinear")
                    {
                        // ROI coordinates, normalized
                        T roi_w_start = rois[roi_idx + 1];
                        T roi_h_start = rois[roi_idx + 2];
                        T roi_w_end = rois[roi_idx + 3];
                        T roi_h_end = rois[roi_idx + 4];

                        T roi_height = (roi_h_end - roi_h_start) * (height - 1);
                        T roi_width = (roi_w_end - roi_w_start) * (width - 1);

                        T roi_height_scale =
                            (pooled_h > 1) ? roi_height / (pooled_h - 1) : static_cast<T>(0);
                        T roi_width_scale =
                            (pooled_w > 1) ? roi_width / (pooled_w - 1) : static_cast<T>(0);

                        for (unsigned int c = 0; c < channels; c++)
                        {
                            for (unsigned int ph = 0; ph < pooled_h; ph++)
                            {
                                for (unsigned int pw = 0; pw < pooled_w; pw++)
                                {
                                    T in_y =
                                        (pooled_h > 1)
                                            ? (ph * roi_height_scale + roi_h_start * (height - 1))
                                            : 0.5 * (roi_h_start + roi_h_end) * (height - 1);
                                    T in_x =
                                        (pooled_w > 1)
                                            ? (pw * roi_width_scale + roi_w_start * (width - 1))
                                            : 0.5 * (roi_w_end + roi_w_start) * (width - 1);

                                    const size_t pool_index =
                                        roi_num * channels * pooled_h * pooled_w +
                                        c * pooled_h * pooled_w + ph * pooled_w + pw;
                                    // Define invalid pooling region to be zero
                                    if (in_y < 0 || in_y > height - 1 || in_x < 0 ||
                                        in_x > width - 1)
                                    {
                                        output[pool_index] = 0;
                                    }
                                    else
                                    {
                                        int top_y_index = static_cast<int>(std::floor(in_y));
                                        int bottom_y_index = static_cast<int>(std::ceil(in_y));
                                        int left_x_index = static_cast<int>(std::floor(in_x));
                                        int right_x_index = static_cast<int>(std::ceil(in_x));

                                        // Clip to input width boundaries
                                        if (right_x_index > width - 1)
                                        {
                                            right_x_index = width - 1;
                                        }

                                        // Clip to input height boundaries
                                        if (bottom_y_index > height - 1)
                                        {
                                            bottom_y_index = height - 1;
                                        }

                                        size_t top_left_idx =
                                            roi_batch_id * channels * height * width +
                                            c * height * width + top_y_index * width + left_x_index;

                                        size_t top_right_idx =
                                            roi_batch_id * channels * height * width +
                                            c * height * width + top_y_index * width +
                                            right_x_index;

                                        size_t bottom_left_idx =
                                            roi_batch_id * channels * height * width +
                                            c * height * width + bottom_y_index * width +
                                            left_x_index;

                                        size_t bottom_right_idx =
                                            roi_batch_id * channels * height * width +
                                            c * height * width + bottom_y_index * width +
                                            right_x_index;

                                        const T top_left = feature_maps[top_left_idx];
                                        const T top_right = feature_maps[top_right_idx];
                                        const T bottom_left = feature_maps[bottom_left_idx];
                                        const T bottom_right = feature_maps[bottom_right_idx];

                                        const T top = top_left + (top_right - top_left) *
                                                                     (in_x - left_x_index);
                                        const T bottom =
                                            bottom_left +
                                            (bottom_right - bottom_left) * (in_x - left_x_index);

                                        output[pool_index] =
                                            top + (bottom - top) * (in_y - top_y_index);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // namespace reference

    } // namespace runtime

} // namespace ngraph
