//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include <string>

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            enum PSROIPoolingMode
            {
                AVG,
                BILINEAR
            };
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
                               int spatial_bins_y)
            {
                PSROIPoolingMode mode;
                if (mode_str == "average")
                {
                    mode = AVG;
                }
                else if (mode_str == "bilinear")
                {
                    mode = BILINEAR;
                }
                else
                {
                    NGRAPH_CHECK(false, "Invalid PS ROI pooling mode");
                }
                int channels_in = input_shape[1];
                int height = input_shape[2];
                int width = input_shape[3];
                int num_rois = output_shape[0];
                int channels_out = output_shape[1];
                int pooling_height = output_shape[2];
                int pooling_width = output_shape[3];
                int num_spatial_bins = spatial_bins_x * spatial_bins_y;
                for (int roi = 0; roi < num_rois; roi++)
                {
                    const T* box = rois + roi * 5;
                    int batch_id = box[0];
                    float start_w = box[1] * spatial_scale;
                    float start_h = box[2] * spatial_scale;
                    float end_w = box[3] * spatial_scale;
                    float end_h = box[4] * spatial_scale;
                    if (mode == AVG)
                    {
                        start_w = std::roundf(start_w);
                        start_h = std::roundf(start_h);
                        end_w = std::roundf(end_w) + 1;
                        end_h = std::roundf(end_h) + 1;
                    }
                    float box_width = end_w - start_w;
                    float box_height = end_h - start_h;
                    float bin_width = box_width / pooling_width;
                    float bin_height = box_height / pooling_height;
                    float width_scale = 0;
                    float height_scale = 0;
                    if (mode == BILINEAR)
                    {
                        bin_width = box_width / spatial_bins_x;
                        bin_height = box_height / spatial_bins_y;
                        if (pooling_width > 1)
                            width_scale = bin_width * (width - 1) / (pooling_width - 1);
                        if (pooling_height > 1)
                            height_scale = bin_height * (height - 1) / (pooling_height - 1);
                    }
                    int c_in = 0;
                    for (int c_out = 0; c_out < channels_out; c_out++)
                    {
                        for (int ph = 0; ph < pooling_height; ph++)
                        {
                            for (int pw = 0; pw < pooling_width; pw++)
                            {
                                int index = ((roi * channels_out + c_out) * pooling_height + ph) *
                                                pooling_width +
                                            pw;
                                output[index] = 0;
                                if (mode == AVG)
                                {
                                    int bin_start_w =
                                        std::min(static_cast<int>(start_w + floorf(pw * bin_width)),
                                                 width - 1);
                                    int bin_start_h = std::min(
                                        static_cast<int>(start_h + floorf(ph * bin_height)),
                                        height - 1);
                                    int current_bin_width =
                                        std::min(
                                            static_cast<int>(start_w + ceilf((pw + 1) * bin_width)),
                                            width) -
                                        bin_start_w;
                                    int current_bin_height =
                                        std::min(static_cast<int>(start_h +
                                                                  ceilf((ph + 1) * bin_height)),
                                                 height) -
                                        bin_start_h;
                                    T sum = 0;
                                    const T* input_offset =
                                        input +
                                        ((batch_id * channels_in + c_in) * height + bin_start_h) *
                                            width +
                                        bin_start_w;
                                    for (int h = 0; h < current_bin_height; h++)
                                    {
                                        for (int w = 0; w < current_bin_width; w++)
                                        {
                                            sum += input_offset[h * width + w];
                                        }
                                    }
                                    output[index] = sum / (current_bin_width * current_bin_height);
                                    c_in++;
                                }
                                else if (mode == BILINEAR)
                                {
                                    c_in = 0;
                                    for (int sby = 0; sby < spatial_bins_y; sby++)
                                    {
                                        for (int sbx = 0; sbx < spatial_bins_x; sbx++)
                                        {
                                            float bin_start_w = start_w + sbx * bin_width;
                                            float bin_start_h = start_h + sby * bin_height;

                                            const T* input_offset = input +
                                                                    (batch_id * channels_in +
                                                                     c_in * channels_out + c_out) *
                                                                        height * width;
                                            float point_x =
                                                pooling_width > 1
                                                    ? (pw * width_scale + bin_start_w * (width - 1))
                                                    : (bin_start_w + bin_start_w + bin_width) *
                                                          (width - 1) / 2;
                                            float point_y =
                                                pooling_height > 1
                                                    ? (ph * height_scale +
                                                       bin_start_h * (height - 1))
                                                    : (bin_start_h + bin_start_h + bin_height) *
                                                          (height - 1) / 2;
                                            if (point_x < width && point_y < height)
                                            {
                                                int left = floorf(point_x);
                                                int right = std::min(
                                                    static_cast<int>(ceilf(point_x)), width - 1);
                                                int top = floorf(point_y);
                                                int bottom = std::min(
                                                    static_cast<int>(ceilf(point_y)), height - 1);
                                                T top_left = input_offset[top * width + left];
                                                T top_right = input_offset[top * width + right];
                                                T bottom_left = input_offset[bottom * width + left];
                                                T bottom_right =
                                                    input_offset[bottom * width + right];

                                                T top_interp =
                                                    top_left +
                                                    (top_right - top_left) * (point_x - left);
                                                T bottom_interp =
                                                    bottom_left +
                                                    (bottom_right - bottom_left) * (point_x - left);
                                                output[index] +=
                                                    top_interp +
                                                    (bottom_interp - top_interp) * (point_y - top);
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
        }
    }
}
