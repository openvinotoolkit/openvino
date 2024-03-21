// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
using ROIPoolingMode = op::v3::ROIAlign::PoolingMode;
using AlignedMode = op::v9::ROIAlign::AlignedMode;

namespace roi_policy {
template <typename T>
struct ROISamplingSpace {
    T minX;
    T minY;
    T height;
    T width;
};

template <typename T>
struct Point {
    T x;
    T y;
};

template <typename T>
class ROIAlignOpDefPolicy {
public:
    ROIAlignOpDefPolicy() : aligned(false), offset_src(0), offset_dst(0) {}

    void Init(const T* rois_, const Shape& shape_, float spatial_scale_, AlignedMode aligned_mode, bool) {
        rois = rois_;
        shape = shape_;
        spatial_scale = spatial_scale_;
        switch (aligned_mode) {
        case AlignedMode::HALF_PIXEL_FOR_NN: {
            aligned = true;
            offset_dst = static_cast<T>(-0.5);
            break;
        }
        case AlignedMode::HALF_PIXEL: {
            aligned = true;
            offset_src = static_cast<T>(0.5);
            offset_dst = static_cast<T>(-0.5);
            break;
        }
        case AlignedMode::ASYMMETRIC: {
            break;
        }
        default: {
            OPENVINO_THROW(std::string("Not supported aligned_mode"));
            break;
        }
        }
    }

    // Sampling space is always a bounding box - for easy iteration over it.
    // However,
    ROISamplingSpace<T> GetSamplingSpaceForIndex(unsigned int index) {
        // Get ROI`s corners
        T x1 = (rois[coordinate_index({index, 0}, shape)] + offset_src) * spatial_scale + offset_dst;
        T y1 = (rois[coordinate_index({index, 1}, shape)] + offset_src) * spatial_scale + offset_dst;
        T x2 = (rois[coordinate_index({index, 2}, shape)] + offset_src) * spatial_scale + offset_dst;
        T y2 = (rois[coordinate_index({index, 3}, shape)] + offset_src) * spatial_scale + offset_dst;

        T roi_width = x2 - x1;
        T roi_height = y2 - y1;

        if (!aligned) {
            roi_width = std::max(roi_width, static_cast<T>(1.0));
            roi_height = std::max(roi_height, static_cast<T>(1.0));
        }

        return {x1, y1, roi_height, roi_width};
    }

    Point<T> Transform(const Point<T>& point) {
        return point;
    }

private:
    const T* rois;
    Shape shape;
    float spatial_scale;
    bool aligned;
    T offset_src;
    T offset_dst;
};

template <typename T>
class ROIAlignRotatedOpDefPolicy {
public:
    ROIAlignRotatedOpDefPolicy() {}

    void Init(const T* rois_, const Shape& shape_, float spatial_scale_, AlignedMode aligned_mode, bool clockwise_) {
        rois = rois_;
        shape = shape_;
        spatial_scale = spatial_scale_;
        clockwise = clockwise_;

        if(aligned_mode != AlignedMode::ASYMMETRIC) {
            OPENVINO_THROW(std::string("ROIAlignRotated: Not supported aligned_mode"));
        }
    }

    // Sampling space is always a bounding box - for easy iteration over it.
    // However,
    ROISamplingSpace<T> GetSamplingSpaceForIndex(unsigned int index) {
        center_x = (rois[coordinate_index({index, 0}, shape)]) * spatial_scale - 0.5f;
        center_y = (rois[coordinate_index({index, 1}, shape)]) * spatial_scale - 0.5f;
        const T width = (rois[coordinate_index({index, 2}, shape)] ) * spatial_scale;
        const T height = (rois[coordinate_index({index, 3}, shape)] ) * spatial_scale;
        T angle = (rois[coordinate_index({index, 4}, shape)]);

        if (clockwise) {
            angle = -angle;
        }
        cos_angle = cos(angle);
        sin_angle = sin(angle);

        const T x1 = -width/2.0f;
        const T y1 = -height/2.0f;

        return {x1, y1, height, width};
    }

    Point<T> Transform(const Point<T>& point) {
        float y = point.y * cos_angle - point.x * sin_angle + center_y;
        float x = point.y * sin_angle + point.x * cos_angle + center_x;
        return {x, y};
    }

private:
    const T* rois;
    Shape shape;
    float spatial_scale;
    bool clockwise;
    float cos_angle;
    float sin_angle;
    T center_x;
    T center_y;
};


};  // namespace roi_policy

template <typename T, template<typename> class TROIPolicy = roi_policy::ROIAlignOpDefPolicy >
void roi_align(const T* feature_maps,
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
               const ROIPoolingMode& pooling_mode,
               const AlignedMode& aligned_mode = AlignedMode::ASYMMETRIC,
               bool clockwise = true) {
    auto C = feature_maps_shape[1];
    auto feature_map_height = feature_maps_shape[2];
    auto feature_map_width = feature_maps_shape[3];
    auto num_rois = rois_shape[0];

    TROIPolicy<T> roi_policy;
    roi_policy.Init(rois, rois_shape, spatial_scale, aligned_mode, clockwise);

    for (unsigned int roi_index = 0; roi_index < num_rois; roi_index++) {
        // Get ROI`s corners
        auto roi_sampling_space = roi_policy.GetSamplingSpaceForIndex(roi_index);

        T x1 = roi_sampling_space.minX;
        T y1 = roi_sampling_space.minY;
        T roi_width = roi_sampling_space.width;
        T roi_height = roi_sampling_space.height;

        T bin_width = roi_width / pooled_width;
        T bin_height = roi_height / pooled_height;

        auto sampling_ratio_x = sampling_ratio == 0 ? static_cast<int>(ceil(bin_width)) : sampling_ratio;
        auto sampling_ratio_y = sampling_ratio == 0 ? static_cast<int>(ceil(bin_height)) : sampling_ratio;

        OPENVINO_ASSERT(sampling_ratio_x >= 0 && sampling_ratio_y >= 0);

        uint64_t num_samples_in_bin = static_cast<uint64_t>(sampling_ratio_x) * static_cast<uint64_t>(sampling_ratio_y);

        T sample_distance_x = bin_width / static_cast<T>(sampling_ratio_x);
        T sample_distance_y = bin_height / static_cast<T>(sampling_ratio_y);

        std::vector<std::pair<unsigned int, unsigned int>> pooling_points;
        std::vector<T> pooling_weights;

        pooling_points.reserve(4 * num_samples_in_bin * pooled_height * pooled_width);
        pooling_weights.reserve(4 * num_samples_in_bin * pooled_height * pooled_width);

        // Save the sample coords and weights as they will be identical across all
        // channels
        for (int y_bin_ind = 0; y_bin_ind < pooled_height; y_bin_ind++) {
            for (int x_bin_ind = 0; x_bin_ind < pooled_width; x_bin_ind++) {
                for (int y_sample_ind = 0; y_sample_ind < sampling_ratio_y; y_sample_ind++) {
                    T pre_sample_y = y1 + static_cast<T>(y_bin_ind) * bin_height +
                                     sample_distance_y * (static_cast<T>(y_sample_ind) + static_cast<T>(0.5f));

                    for (int64_t x_sample_ind = 0; x_sample_ind < sampling_ratio_x; x_sample_ind++) {
                        T pre_sample_x = x1 + static_cast<T>(x_bin_ind) * bin_width +
                                         sample_distance_x * (static_cast<T>(x_sample_ind) + static_cast<T>(0.5f));

                        if (pre_sample_x < -1.0 || pre_sample_x > static_cast<T>(feature_map_width) ||
                            pre_sample_y < -1.0 || pre_sample_y > static_cast<T>(feature_map_height)) {
                            // For this sample we save 4x point (0,0) with weight 0
                            pooling_points.insert(pooling_points.end(), 4, {0, 0});
                            pooling_weights.insert(pooling_weights.end(), 4, T{0});
                            continue;
                        }

                        // Transform pre_sample_x and pre_sample_ y
                        auto transformed = roi_policy.Transform({pre_sample_x, pre_sample_y});

                        T sample_x = transformed.x;
                        T sample_y = transformed.y;

                        sample_x = std::max(sample_x, T{0});
                        sample_y = std::max(sample_y, T{0});

                        auto sample_y_low = static_cast<unsigned int>(sample_y);
                        auto sample_x_low = static_cast<unsigned int>(sample_x);
                        unsigned int sample_y_high;
                        unsigned int sample_x_high;

                        if (sample_y_low >= feature_map_height - 1) {
                            sample_y_high = sample_y_low = static_cast<unsigned int>(feature_map_height - 1);
                            sample_y = static_cast<T>(sample_y_low);
                        } else {
                            sample_y_high = sample_y_low + 1;
                        }

                        if (sample_x_low >= feature_map_width - 1) {
                            sample_x_high = sample_x_low = static_cast<unsigned int>(feature_map_width - 1);
                            sample_x = static_cast<T>(sample_x_low);
                        } else {
                            sample_x_high = sample_x_low + 1;
                        }
                        pooling_points.push_back({sample_y_low, sample_x_low});
                        pooling_points.push_back({sample_y_low, sample_x_high});
                        pooling_points.push_back({sample_y_high, sample_x_low});
                        pooling_points.push_back({sample_y_high, sample_x_high});

                        // weight calculation for bilinear interpolation
                        auto ly = sample_y - static_cast<T>(sample_y_low);
                        auto lx = sample_x - static_cast<T>(sample_x_low);
                        auto hy = static_cast<T>(1.) - ly;
                        auto hx = static_cast<T>(1.) - lx;

                        pooling_weights.push_back(hy * hx);
                        pooling_weights.push_back(hy * lx);
                        pooling_weights.push_back(ly * hx);
                        pooling_weights.push_back(ly * lx);
                    }
                }
            }
        }

        std::vector<T> tmp_out;

        for (unsigned int channel_index = 0; channel_index < C; channel_index++) {
            tmp_out.reserve(pooled_height * pooled_width);
            unsigned int sample_index = 0;
            for (int y_bin_ind = 0; y_bin_ind < pooled_height; y_bin_ind++) {
                for (int x_bin_ind = 0; x_bin_ind < pooled_width; x_bin_ind++) {
                    T pooled_value = 0;
                    for (unsigned int bin_sample_ind = 0; bin_sample_ind < num_samples_in_bin; bin_sample_ind++) {
                        // the four parts are values of the four closest surrounding
                        // neighbours of considered sample, then basing on all sampled
                        // values in bin we calculate pooled value
                        const auto batch_index = static_cast<size_t>(batch_indices[roi_index]);
                        auto sample_part_1 = feature_maps[coordinate_index({batch_index,
                                                                            channel_index,
                                                                            pooling_points[sample_index].first,
                                                                            pooling_points[sample_index].second},
                                                                           feature_maps_shape)];
                        auto sample_part_2 = feature_maps[coordinate_index({batch_index,
                                                                            channel_index,
                                                                            pooling_points[sample_index + 1].first,
                                                                            pooling_points[sample_index + 1].second},
                                                                           feature_maps_shape)];
                        auto sample_part_3 = feature_maps[coordinate_index({batch_index,
                                                                            channel_index,
                                                                            pooling_points[sample_index + 2].first,
                                                                            pooling_points[sample_index + 2].second},
                                                                           feature_maps_shape)];
                        auto sample_part_4 = feature_maps[coordinate_index({batch_index,
                                                                            channel_index,
                                                                            pooling_points[sample_index + 3].first,
                                                                            pooling_points[sample_index + 3].second},
                                                                           feature_maps_shape)];

                        T sample_value = pooling_weights[sample_index] * sample_part_1 +
                                         pooling_weights[sample_index + 1] * sample_part_2 +
                                         pooling_weights[sample_index + 2] * sample_part_3 +
                                         pooling_weights[sample_index + 3] * sample_part_4;
                        switch (pooling_mode) {
                        case ROIPoolingMode::MAX: {
                            pooled_value = sample_value > pooled_value ? sample_value : pooled_value;
                            break;
                        }
                        case ROIPoolingMode::AVG:
                        default: {
                            pooled_value += sample_value / (num_samples_in_bin);
                        }
                        }
                        sample_index += 4;
                    }
                    tmp_out.push_back(pooled_value);
                }
            }
            // save the calculations for all bins across this channel
            auto output_channel_offset = coordinate_index({roi_index, channel_index, 0ul, 0ul}, out_shape);
            std::copy(tmp_out.begin(), tmp_out.end(), out + output_channel_offset);

            tmp_out.clear();
        }
    }
}
}  // namespace reference
}  // namespace ov
