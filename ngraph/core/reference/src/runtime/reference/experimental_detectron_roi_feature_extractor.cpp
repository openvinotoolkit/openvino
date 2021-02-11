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

#include "ngraph/runtime/reference/experimental_detectron_roi_feature_extractor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <queue>
#include <vector>
#include "ngraph/op/experimental_detectron_roi_feature.hpp"
#include "ngraph/shape.hpp"

namespace
{
    void redistribute_rois(const float* rois,
                           int64_t* level_ids,
                           const int64_t num_rois,
                           const int64_t levels_num)
    {
        const float canonical_scale = 224.0f;
        const int64_t canonical_level = 2;

        for (int64_t i = 0; i < num_rois; ++i)
        {
            const float x0 = rois[4 * i + 0];
            const float y0 = rois[4 * i + 1];
            const float x1 = rois[4 * i + 2];
            const float y1 = rois[4 * i + 3];

            int64_t target_level = levels_num;
            float area = (x1 - x0) * (y1 - y0);
            if (area > 0)
            {
                area = std::sqrt(area) / canonical_scale;
                area = std::log2(area + 1e-6f);
                target_level = static_cast<int64_t>(std::floor(area + canonical_level));
                target_level = (std::max)(0, (std::min)(levels_num - 1, target_level));
            }

            level_ids[i] = target_level;
        }
    }

    void reorder(const float* src_data,
                 const int64_t* ranks,
                 const int64_t n,
                 const int64_t step,
                 float* dst_data,
                 int64_t* dst_mapping)
    {
        std::iota(dst_mapping, dst_mapping + n, 0);
        std::sort(dst_mapping, dst_mapping + n, [&ranks](size_t i1, size_t i2) {
            return ranks[i1] < ranks[i2];
        });

        for (int64_t i = 0; i < n; ++i)
        {
            const int64_t j = dst_mapping[i];
            assert(0 <= j && j < n);
            memcpy(dst_data + i * step, src_data + j * step, sizeof(float) * step);
        }
    }

    void split_points(const std::vector<int64_t>& ids,
                      std::vector<int64_t>& rois_per_level,
                      const int64_t levels_num)
    {
        rois_per_level.clear();
        rois_per_level.resize(levels_num, 0);
        for (size_t i = 0; i < ids.size(); ++i)
        {
            assert(0 <= ids[i] && ids[i] < levels_num);
            rois_per_level[ids[i]]++;
        }
        for (int64_t i = 1; i < levels_num; ++i)
        {
            rois_per_level[i] += rois_per_level[i - 1];
        }
        rois_per_level.insert(rois_per_level.begin(), 0);
    }

    // implementation taken from Caffe2
    template <typename T>
    struct PreCalc
    {
        int64_t pos1;
        int64_t pos2;
        int64_t pos3;
        int64_t pos4;
        T w1;
        T w2;
        T w3;
        T w4;
    };

    template <typename T>
    void pre_calc_for_bilinear_interpolate(const int64_t height,
                                           const int64_t width,
                                           const int64_t pooled_height,
                                           const int64_t pooled_width,
                                           const int64_t iy_upper,
                                           const int64_t ix_upper,
                                           T roi_start_h,
                                           T roi_start_w,
                                           T bin_size_h,
                                           T bin_size_w,
                                           int64_t roi_bin_grid_h,
                                           int64_t roi_bin_grid_w,
                                           std::vector<PreCalc<T>>& pre_calc)
    {
        int64_t pre_calc_index = 0;
        for (int64_t ph = 0; ph < pooled_height; ph++)
        {
            for (int64_t pw = 0; pw < pooled_width; pw++)
            {
                for (int64_t iy = 0; iy < iy_upper; iy++)
                {
                    const T yy = roi_start_h + ph * bin_size_h +
                                 static_cast<T>(iy + 0.5f) * bin_size_h /
                                     static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
                    for (int64_t ix = 0; ix < ix_upper; ix++)
                    {
                        const T xx =
                            roi_start_w + pw * bin_size_w +
                            static_cast<T>(ix + 0.5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                        T x = xx;
                        T y = yy;
                        // deal with: inverse elements are out of feature map boundary
                        if (y < -1.0 || y > height || x < -1.0 || x > width)
                        {
                            // empty
                            PreCalc<T> pc;
                            pc.pos1 = 0;
                            pc.pos2 = 0;
                            pc.pos3 = 0;
                            pc.pos4 = 0;
                            pc.w1 = 0;
                            pc.w2 = 0;
                            pc.w3 = 0;
                            pc.w4 = 0;
                            pre_calc.at(pre_calc_index) = pc;
                            pre_calc_index += 1;
                            continue;
                        }

                        if (y <= 0)
                        {
                            y = 0;
                        }
                        if (x <= 0)
                        {
                            x = 0;
                        }

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = 0;
                        int64_t x_high = 0;

                        if (y_low >= height - 1)
                        {
                            y_high = y_low = height - 1;
                            y = (T)y_low;
                        }
                        else
                        {
                            y_high = y_low + 1;
                        }

                        if (x_low >= width - 1)
                        {
                            x_high = x_low = width - 1;
                            x = (T)x_low;
                        }
                        else
                        {
                            x_high = x_low + 1;
                        }

                        T ly = y - y_low;
                        T lx = x - x_low;
                        T hy = static_cast<T>(1) - ly, hx = static_cast<T>(1) - lx;
                        T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                        // save weights and indeces
                        PreCalc<T> pc;
                        pc.pos1 = y_low * width + x_low;
                        pc.pos2 = y_low * width + x_high;
                        pc.pos3 = y_high * width + x_low;
                        pc.pos4 = y_high * width + x_high;
                        pc.w1 = w1;
                        pc.w2 = w2;
                        pc.w3 = w3;
                        pc.w4 = w4;
                        pre_calc[pre_calc_index] = pc;

                        pre_calc_index += 1;
                    }
                }
            }
        }
    }


    template <typename T>
    void ROIAlignForward_kernel(const int64_t level_rois_num,
                                const T* bottom_data,
                                const T& spatial_scale,
                                const int64_t channels,
                                const int64_t height,
                                const int64_t width,
                                const int64_t pooled_height,
                                const int64_t pooled_width,
                                const int64_t sampling_ratio,
                                const T* bottom_rois,
                                const bool aligned,
                                T* top_data)
    {
        int64_t roi_cols = 4;
        // (n, c, ph, pw) is an element in the pooled output
        for (int64_t n = 0; n < level_rois_num; ++n)
        {
            int64_t index_n = n * channels * pooled_width * pooled_height;
            const T* offset_bottom_rois = bottom_rois + n * roi_cols;
            int64_t roi_batch_ind = 0;

            T offset = aligned ? (T)0.5 : (T)0.0;
            // Do not using rounding; this implementation detail is critical
            T roi_start_w = offset_bottom_rois[0] * spatial_scale - offset;
            T roi_start_h = offset_bottom_rois[1] * spatial_scale - offset;
            T roi_end_w = offset_bottom_rois[2] * spatial_scale - offset;
            T roi_end_h = offset_bottom_rois[3] * spatial_scale - offset;

            // Force malformed ROIs to be 1x1
            T roi_width = (std::max)(roi_end_w - roi_start_w, (T)1.0);
            T roi_height = (std::max)(roi_end_h - roi_start_h, (T)1.0);
            T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

            // We use roi_bin_grid to sample the grid and mimic integral
            int64_t roi_bin_grid_h =
                (sampling_ratio > 0)
                    ? sampling_ratio
                    : static_cast<int64_t>(ceil(roi_height / pooled_height)); // e.g., = 2
            int64_t roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio
                                     : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

            // We do average (integral) pooling inside a bin
            const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w); // e.g. = 4

            // we want to precalculate indeces and weights shared by all chanels,
            // this is the key point of optimiation
            std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width *
                                             pooled_height);
            pre_calc_for_bilinear_interpolate(height,
                                              width,
                                              pooled_height,
                                              pooled_width,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              roi_start_h,
                                              roi_start_w,
                                              bin_size_h,
                                              bin_size_w,
                                              roi_bin_grid_h,
                                              roi_bin_grid_w,
                                              pre_calc);

            for (int64_t c = 0; c < channels; c++)
            {
                int64_t index_n_c = index_n + c * pooled_width * pooled_height;
                const T* offset_bottom_data =
                    bottom_data + (roi_batch_ind * channels + c) * height * width;
                int64_t pre_calc_index = 0;

                for (int64_t ph = 0; ph < pooled_height; ph++)
                {
                    for (int64_t pw = 0; pw < pooled_width; pw++)
                    {
                        int64_t index = index_n_c + ph * pooled_width + pw;

                        T output_val = 0.;
                        for (int iy = 0; iy < roi_bin_grid_h; iy++)
                        {
                            for (int ix = 0; ix < roi_bin_grid_w; ix++)
                            {
                                PreCalc<T> pc = pre_calc[pre_calc_index];
                                output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                                              pc.w2 * offset_bottom_data[pc.pos2] +
                                              pc.w3 * offset_bottom_data[pc.pos3] +
                                              pc.w4 * offset_bottom_data[pc.pos4];

                                pre_calc_index += 1;
                            }
                        }
                        output_val /= count;

                        top_data[index] = output_val;
                    }  // for pw
                }      // for ph
            }          // for c
        }
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void experimental_detectron_roi_feature_extractor(
                const std::vector<float*>& inputs,
                const std::vector<Shape>& input_shapes,
                const op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs,
                float* output_rois_features,
                float* output_rois)
            {
                const int64_t output_dim = attrs.output_size;
                const int64_t pooled_height = output_dim;
                const int64_t pooled_width = output_dim;
                const auto& pyramid_scales = attrs.pyramid_scales;
                const int64_t sampling_ratio = attrs.sampling_ratio;
                const bool aligned = attrs.aligned;

                const int64_t levels_num = static_cast<int64_t>(inputs.size() - 1);
                const int64_t num_rois = static_cast<int64_t>(input_shapes[0]);
                const int64_t channels_num = static_cast<int64_t>(input_shapes[1]);
                const int64_t feaxels_per_roi = pooled_height * pooled_width * channels_num;

                const float* input_rois = inputs[0];
                std::vector<int64_t> level_ids(num_rois, 0);
                redistribute_rois(input_rois, level_ids.data(), num_rois, levels_num);

                std::vector<float> reordered_rois(4 * num_rois, 0);
                std::vector<int64_t> original_rois_mapping(num_rois, 0);
                reorder(input_rois,
                        level_ids.data(),
                        num_rois,
                        4,
                        reordered_rois.data(),
                        original_rois_mapping.data());

                std::vector<int64_t> rois_per_level;
                split_points(level_ids, rois_per_level, levels_num + 1);

                std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
                for (int64_t i = 0; i < levels_num; ++i)
                {
                    const int64_t level_rois_offset = rois_per_level[i];
                    const int64_t level_rois_num = rois_per_level[i + 1] - level_rois_offset;
                    if (level_rois_num > 0)
                    {
                        const float* featuremap = inputs[1 + i];
                        const int64_t featuremap_height =
                            static_cast<int64_t>(input_shapes[1 + i][2]);
                        const int64_t featuremap_width =
                            static_cast<int64_t>(input_shapes[1 + i][3]);
                        ROIAlignForward_kernel<float>(
                            level_rois_num,
                            featuremap,
                            1.0f / pyramid_scales[i],
                            channels_num,
                            featuremap_height,
                            featuremap_width,
                            pooled_height,
                            pooled_width,
                            sampling_ratio,
                            &reordered_rois[4 * level_rois_offset],
                            aligned,
                            &output_rois_features_temp[feaxels_per_roi * level_rois_offset]);
                    }
                }

                std::vector<int> dummy_mapping(num_rois, 0);
                reorder(output_rois_features_temp.data(),
                        original_rois_mapping.data(),
                        num_rois,
                        feaxels_per_roi,
                        output_rois_features,
                        dummy_mapping.data());
                memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
            }

            void experimental_detectron_roi_feature_extractor_postprocessing(
                const HostTensorVector& outputs,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_roi_features,
                const std::vector<float>& output_rois,
                const Shape& output_roi_features_shape,
                const Shape& output_rois_shape)
            {
            }
        }
    }
}
