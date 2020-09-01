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
                                         const int pooled_width,
                                         const int pooled_height,
                                         const int sampling_ratio,
                                         const float spatial_scale,
                                         const ROIPoolingMode& pooling_mode)
            {
                auto N = feature_maps_shape[0];
                auto C = feature_maps_shape[1];
                auto H = feature_maps_shape[2];
                auto W = feature_maps_shape[3];
                auto num_rois = rois_shape[0];

                CoordinateTransform rois_transform(rois_shape);

                auto x1 = rois[rois_transform.index({0, 0})] * spatial_scale;
                auto y1 = rois[rois_transform.index({0, 1})] * spatial_scale;
                auto x2 = rois[rois_transform.index({0, 2})] * spatial_scale;
                auto y2 = rois[rois_transform.index({0, 3})] * spatial_scale;
                auto roi_w = fmax(x2 - x1, static_cast<T>(1));
                auto roi_h = fmax(y2 - y1, static_cast<T>(1));

                auto bin_w = roi_w / pooled_width;
                auto bin_h = roi_h / pooled_height;

                auto sampling_w = sampling_ratio * pooled_width;
                auto sampling_h = sampling_ratio * pooled_height;

                auto sampling_x1 = x1 + 0.5 * bin_w / sampling_ratio;
                auto sampling_y1 = y1 + 0.5 * bin_h / sampling_ratio;
                auto sampling_x2 = x2 - 0.5 * bin_w / sampling_ratio;
                auto sampling_y2 = y2 - 0.5 * bin_h / sampling_ratio;

                // switch (pooling_mode)
                // {
                // case (ROIPoolingMode::MAX):
                // {
                //     out = max_pool(samples, Shape{sampling_ratio, sampling_ratio});
                // }
                // case (ROIPoolingMode::AVG):
                // {
                //     out = avg_pool(samples, Shape{sampling_ratio, sampling_ratio});
                // }
                // }

                return;
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph