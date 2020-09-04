//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cmath>
#include <iostream>
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/roi_align.hpp" // for ROIAlign:PoolingMode
#include "ngraph/shape.hpp"
namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            using ROIPoolingMode = op::v3::ROIAlign::PoolingMode;
            template <typename T>
            static inline void roi_align(const T* feature_maps,
                                         const T* rois,
                                         const int64_t* batch_indices,
                                         T* out,
                                         const Shape& feature_maps_shape,
                                         const Shape& rois_shape,
                                         const Shape& batch_indices_shape,
                                         const Shape& out_shape,
                                         const int pooled_height,
                                         const int pooled_width,
                                         const int sampling_ratio,
                                         const float spatial_scale,
                                         const ROIPoolingMode& pooling_mode)
            {
                auto N = feature_maps_shape[0];
                auto C = feature_maps_shape[1];
                auto H = feature_maps_shape[2];
                auto W = feature_maps_shape[3];
                auto num_rois = rois_shape[0];

                CoordinateTransform feature_maps_transform(feature_maps_shape);
                CoordinateTransform rois_transform(rois_shape);
                CoordinateTransform out_transform(out_shape);

                for (uint64_t roi_index = 0; roi_index < num_rois; roi_index++)
                {
                    // Get ROI`s corners
                    T x1 = rois[rois_transform.index({roi_index, 0})] * spatial_scale;
                    T y1 = rois[rois_transform.index({roi_index, 1})] * spatial_scale;
                    T x2 = rois[rois_transform.index({roi_index, 2})] * spatial_scale;
                    T y2 = rois[rois_transform.index({roi_index, 3})] * spatial_scale;

                    T roi_w = fmax(x2 - x1, static_cast<T>(1));
                    auto roi_h = fmax(y2 - y1, static_cast<T>(1));

                    // W and H of each bin- already relative to spatial scale
                    T bin_w = roi_w / pooled_width;
                    T bin_h = roi_h / pooled_height;

                    auto sample_count_horizontal = sampling_ratio * pooled_width;
                    auto sample_count_vertical = sampling_ratio * pooled_height;

                    T sample_distance_horizontal = bin_w / static_cast<T>(sampling_ratio + 1);
                    T sample_distance_vertical = bin_h / static_cast<T>(sampling_ratio + 1);

                    // Prepare coordinates for 4 pooling points for each of the sampling points of
                    // every bin
                    std::vector<std::pair<uint64_t, uint64_t>> pooling_points;
                    std::vector<T> pooling_weights;
                    pooling_points.reserve(4 * sample_count_horizontal * sample_count_vertical);
                    pooling_weights.reserve(4 * sample_count_horizontal * sample_count_vertical);

                    for (int64_t bin_vertical = 0; bin_vertical < pooled_height; bin_vertical++)
                    {
                        for (int64_t bin_horizontal = 0; bin_horizontal < pooled_width;
                             bin_horizontal++)
                        {
                            for (int64_t i = 0; i < sampling_ratio; i++)
                            {
                                T sample_y = y1 + bin_vertical * bin_h +
                                                sample_distance_vertical *
                                                    ((static_cast<T>(i)) + static_cast<T>(1.0f));

                                for (int64_t j = 0; j < sampling_ratio; j++)
                                {
                                    T sample_x =
                                        x1 + bin_horizontal * bin_w +
                                        sample_distance_horizontal *
                                            ((static_cast<T>(j)) + static_cast<T>(1.0f));
                                    // for each sampling point we have 4 coordinate pairs, that
                                    // address pooled values
                                    std::cout << "bin [" << bin_vertical << " ," << bin_horizontal
                                              << "] sample [" << i << ", " << j << "] at ["
                                              << sample_y << ", " << sample_x << "]" << std::endl;
                                    if (sample_x < -1.0 || sample_x > W || sample_y < -1.0 ||
                                        sample_y > H)
                                    {
                                        pooling_points.push_back({0, 0});
                                        pooling_points.push_back({0, 0});
                                        pooling_points.push_back({0, 0});
                                        pooling_points.push_back({0, 0});

                                        pooling_weights.push_back(0);
                                        pooling_weights.push_back(0);
                                        pooling_weights.push_back(0);
                                        pooling_weights.push_back(0);

                                        continue;
                                    }
                                    if (sample_x < 0.0)
                                    {
                                        sample_x = 0.0;
                                    }
                                    if (sample_y < 0.0)
                                    {
                                        sample_y = 0.0;
                                    }

                                    auto sample_y_low = static_cast<uint64_t>(sample_y);
                                    auto sample_x_low = static_cast<uint64_t>(sample_x);
                                    uint64_t sample_y_high;
                                    uint64_t sample_x_high;

                                    if (sample_y_low >= H - 1)
                                    {
                                        sample_y_high = sample_y_low = H - 1;
                                        sample_y = (T)sample_y_low;
                                    }
                                    else
                                    {
                                        sample_y_high = sample_y_low + 1;
                                    }

                                    if (sample_x_low >= H - 1)
                                    {
                                        sample_x_high = sample_x_low = W - 1;
                                        sample_x = (T)sample_x_low;
                                    }
                                    else
                                    {
                                        sample_x_high = sample_x_low + 1;
                                    }

                                    T ly = sample_y - sample_y_low;
                                    T lx = sample_x - sample_x_low;
                                    T hy = static_cast<T>(1.) - ly;
                                    T hx = static_cast<T>(1.) - lx;

                                    pooling_points.push_back({sample_y_low, sample_x_low});
                                    pooling_points.push_back({sample_y_low, sample_x_high});
                                    pooling_points.push_back({sample_y_high, sample_x_low});
                                    pooling_points.push_back({sample_y_high, sample_x_high});

                                    pooling_weights.push_back(hy * hx);
                                    pooling_weights.push_back(hy * lx);
                                    pooling_weights.push_back(ly * hx);
                                    pooling_weights.push_back(ly * lx);
                                }
                            }
                        }
                    }

                    for (uint64_t channel_index = 0; channel_index < C; channel_index++)
                    {
                        // Go over each bin
                        for (int64_t bin_vertical = 0; bin_vertical < pooled_height; bin_vertical++)
                        {
                            for (int64_t bin_horizontal = 0; bin_horizontal < pooled_width;
                                 bin_horizontal++)
                            {
                                T pooled_value = 0;
                                for (int64_t sample_in_bin_index = 0;
                                     sample_in_bin_index < sampling_ratio * sampling_ratio;
                                     sample_in_bin_index++)
                                {
                                    auto sample_index =
                                        4 * (sample_in_bin_index +
                                             (bin_horizontal + bin_vertical * pooled_width) *
                                                 sampling_ratio * sampling_ratio);

                                    auto s1 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index].first,
                                         pooling_points[sample_index].second})];
                                    auto s2 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 1].first,
                                         pooling_points[sample_index + 1].second})];
                                    auto s3 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 2].first,
                                         pooling_points[sample_index + 2].second})];
                                    auto s4 = feature_maps[feature_maps_transform.index(
                                        {static_cast<uint64_t>(batch_indices[roi_index]),
                                         channel_index,
                                         pooling_points[sample_index + 3].first,
                                         pooling_points[sample_index + 3].second})];
                                    auto sample_value = pooling_weights[sample_index] * s1 +
                                                        pooling_weights[sample_index + 1] * s2 +
                                                        pooling_weights[sample_index + 2] * s3 +
                                                        pooling_weights[sample_index + 3] * s4;
                                    switch (pooling_mode)
                                    {
                                    case ROIPoolingMode::MAX:
                                    {
                                        pooled_value = sample_value > pooled_value ? sample_value
                                                                                   : pooled_value;
                                        break;
                                    }
                                    default:
                                    {
                                        pooled_value +=
                                            sample_value / (sampling_ratio * sampling_ratio);
                                    }
                                    }
                                    // TODO: when this works, save all output for a single bin at
                                    // one time
                                }
                                std::cout << static_cast<float>(pooled_value) << " ";
                                memcpy(&out[out_transform.index(
                                           {static_cast<uint64_t>(roi_index),
                                            static_cast<uint64_t>(channel_index),
                                            static_cast<uint64_t>(bin_vertical),
                                            static_cast<uint64_t>(bin_horizontal)})],
                                       &pooled_value,
                                       sizeof(T));
                            }
                            std::cout << "\n";
                        }
                    }
                }
                return;
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
