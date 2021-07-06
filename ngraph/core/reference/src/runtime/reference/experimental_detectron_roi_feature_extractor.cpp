// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/experimental_detectron_roi_feature_extractor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include "ngraph/op/experimental_detectron_roi_feature.hpp"
#include "ngraph/shape.hpp"

namespace
{
    constexpr int64_t input_rois_port = 0;
    constexpr int64_t input_features_start_port = 1;

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
                target_level =
                    std::max(static_cast<int64_t>(0), std::min(levels_num - 1, target_level));
            }

            level_ids[i] = target_level;
        }
    }

    void reord(const float* src_data,
               const int64_t* ranks,
               const int64_t n,
               const int64_t step,
               float* dst_data,
               int64_t* dst_mapping)
    {
        std::iota(dst_mapping, dst_mapping + n, static_cast<int64_t>(0));
        std::sort(dst_mapping, dst_mapping + n, [&ranks](int64_t i1, int64_t i2) {
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
        for (int64_t id : ids)
        {
            assert(0 <= id && id < levels_num);
            rois_per_level[id]++;
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
        std::cout << "            pre_calc_index: " << pre_calc_index << "\n";
        std::cout << "            Loop with respect to ph in range(0, pooled_height)\n";
        for (int64_t ph = 0; ph < pooled_height; ph++)
        {
            std::cout << "                ph: " << ph << "\n";
            std::cout << "                Loop with respect to pw in range(0, pooled_width)\n";
            for (int64_t pw = 0; pw < pooled_width; pw++)
            {
                std::cout << "                    pw: " << pw << "\n";
                std::cout << "                    Loop with respect to iy in range(0, iy_upper)\n";
                for (int64_t iy = 0; iy < iy_upper; iy++)
                {
                    std::cout << "                        iy: " << iy << "\n";
                    const T yy = roi_start_h + static_cast<T>(ph) * bin_size_h +
                                 static_cast<T>(static_cast<T>(iy) + 0.5f) * bin_size_h /
                                     static_cast<T>(roi_bin_grid_h);
                    std::cout << "                        yy: " << yy << "\n";
                    std::cout << "                        Loop with respect to ix in range(0, ix_upper)\n";
                    for (int64_t ix = 0; ix < ix_upper; ix++)
                    {
                        std::cout << "                            ix: " << ix << "\n";
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

//                         y = std::max(y, static_cast<T>(0.0));
                        if (y <= 0)
                        {
                            y = 0;
                        }
//                         x = std::max(x, static_cast<T>(0.0));
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
//                             y = static_cast<T>(y_low);
                            y = (T)y_low;
                        }
                        else
                        {
                            y_high = y_low + 1;
                        }

                        if (x_low >= width - 1)
                        {
                            x_high = x_low = width - 1;
//                             x = static_cast<T>(x_low);
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
    void ROIAlignForward_cpu_kernel(const int64_t nthreads,
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
        std::cout << "    ROIAlignForward_cpu_kernel running...\n";
        std::cout << "    Its arguments are\n";
        std::cout << "    nthreads:       " << nthreads << "\n";
        std::cout << "    bottom_data:    " << bottom_data << "\n";
        std::cout << "    spatial_scale:  " << spatial_scale << "\n";
        std::cout << "    channels:       " << channels << "\n";
        std::cout << "    height:         " << height << "\n";
        std::cout << "    width:          " << width << "\n";
        std::cout << "    pooled_height:  " << pooled_height << "\n";
        std::cout << "    pooled_width:   " << pooled_width << "\n";
        std::cout << "    sampling_ratio: " << sampling_ratio << "\n";
        std::cout << "    bottom_rois:    " << bottom_rois << "\n";
        std::cout << "    aligned:        " << (aligned ? "true" : "false") << "\n\n";
        std::cout << "    top_data:       " << top_data << "\n";
        int64_t roi_cols = 4;

        int64_t n_rois = nthreads / channels / pooled_width / pooled_height;
        std::cout << "    roi_cols:       " << roi_cols << "\n";
        std::cout << "    n_rois:         " << n_rois << "\n";
        // (n, c, ph, pw) is an element in the pooled output
        std::cout << "Loop with respect to n in range(0, n_rois)...\n";
        for (int64_t n = 0; n < n_rois; ++n)
        {
            std::cout << "        n:       " << n << "\n";
            int64_t index_n = n * channels * pooled_width * pooled_height;
            std::cout << "        index_n: " << index_n << "\n";

            // roi could have 4 or 5 columns
            const T* offset_bottom_rois = bottom_rois + n * roi_cols;
            int64_t roi_batch_ind = 0;
            std::cout << "        offset_bottom_rois before if stmt: " << offset_bottom_rois << "\n";
            std::cout << "        roi_batch_ind before if stmt:      " << roi_batch_ind << "\n";
            if (roi_cols == 5)
            {
                roi_batch_ind = static_cast<int64_t>(offset_bottom_rois[0]);
                offset_bottom_rois++;
            }
            std::cout << "        offset_bottom_rois after if stmt:  " << offset_bottom_rois << "\n";
            std::cout << "        roi_batch_ind after if stmt:       " << roi_batch_ind << "\n";

//             T offset = aligned ? static_cast<T>(0.5) : static_cast<T>(0.0);
            T offset = aligned ? (T)0.5 : (T)0.0;
            std::cout << "        offset: " << offset << "\n";
            // Do not using rounding; this implementation detail is critical
            T roi_start_w = offset_bottom_rois[0] * spatial_scale - offset;
            T roi_start_h = offset_bottom_rois[1] * spatial_scale - offset;
            T roi_end_w = offset_bottom_rois[2] * spatial_scale - offset;
            T roi_end_h = offset_bottom_rois[3] * spatial_scale - offset;
            std::cout << "        roi_start_w: " << roi_start_w << "\n";
            std::cout << "        roi_start_h: " << roi_start_h << "\n";
            std::cout << "        roi_end_w:   " << roi_end_w << "\n";
            std::cout << "        roi_end_h:   " << roi_end_h << "\n";

            // Force malformed ROIs to be 1x1
//             T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(1.0));
            T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
//             T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(1.0));
            T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
            T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
            T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);
            std::cout << "        roi_width:   " << roi_width << "\n";
            std::cout << "        roi_height:  " << roi_height << "\n";
            std::cout << "        bin_size_h:  " << bin_size_h << "\n";
            std::cout << "        bin_size_w:  " << bin_size_w << "\n";

            // We use roi_bin_grid to sample the grid and mimic integral
            int64_t roi_bin_grid_h =
                (sampling_ratio > 0)
                    ? sampling_ratio
                    : static_cast<int64_t>(std::ceil(roi_height / pooled_height)); // e.g., = 2
            int64_t roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio
                                     : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

            // We do average (integral) pooling inside a bin
            const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w); // e.g. = 4
            std::cout << "        roi_bin_grid_h:  " << roi_bin_grid_h << "\n";
            std::cout << "        roi_bin_grid_w:  " << roi_bin_grid_w << "\n";
            std::cout << "        count:           " << count << "\n";

            // we want to precalculate indices and weights shared by all channels,
            // this is the key point of optimization
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
            std::cout << "        pre_calc:\n";
            for (const auto& p : pre_calc)
            {
                std::cout << "            PreCalc{"
                          << ".pos1=" << p.pos1 << ", "
                          << ".pos2=" << p.pos2 << ", "
                          << ".pos3=" << p.pos3 << ", "
                          << ".pos4=" << p.pos4 << ", "
                          << ".w1=" << p.w1 << ", "
                          << ".w2=" << p.w2 << ", "
                          << ".w3=" << p.w3 << ", "
                          << ".w4=" << p.w4 << "}\n";
            }
            std::cout << "\n\n";

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
                        for (int64_t iy = 0; iy < roi_bin_grid_h; iy++)
                        {
                            for (int64_t ix = 0; ix < roi_bin_grid_w; ix++)
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
                    } // for pw
                }     // for ph
            }         // for c
        }
    }
} // namespace

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void experimental_detectron_roi_feature_extractor(
                const std::vector<std::vector<float>>& inputs,
                const std::vector<Shape>& input_shapes,
                const op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs,
                float* output_rois_features,
                float* output_rois)
            {
                std::cout << "Running reference implementation calculations for "
                             "ExperimentalDetectronROIFeatureExtractor...\n";
                std::cout << "Its arguments are\n\n";
                std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
                std::cout << "Input data:\n";
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    std::cout << "    input number " << i << " has data: ";
                    for (auto x : inputs[i])
                    {
                        std::cout << x << ", ";
                    }
                    std::cout << "\n\n";
                }
                std::cout << "Input shapes:\n";
                for (size_t i = 0; i < input_shapes.size(); ++i)
                {
                    std::cout << "    input number " << i << " has shape: " << input_shapes[i] <<"\n\n";
                }

                int64_t output_dim_ = attrs.output_size;
                auto pyramid_scales_ = attrs.pyramid_scales;
                int64_t sampling_ratio_ = attrs.sampling_ratio;
                bool aligned_ = attrs.aligned;
                int64_t pooled_height_ = output_dim_;
                int64_t pooled_width_ = output_dim_;

                const int64_t levels_num =
                    static_cast<int64_t>(inputs.size() - input_features_start_port);
                const int64_t num_rois = static_cast<int64_t>(input_shapes[input_rois_port][0]);
                const int64_t channels_num =
                    static_cast<int64_t>(input_shapes[input_features_start_port][1]);
                const int64_t feaxels_per_roi = pooled_height_ * pooled_width_ * channels_num;

                std::cout << "output_dim_:     " << output_dim_ << "\n";
                std::cout << "pyramid_scales_: ";
                for (auto x : pyramid_scales_)
                {
                    std::cout << x << ", ";
                }
                std::cout << "\n";
                std::cout << "sampling_ratio_: " << sampling_ratio_ << "\n";
                std::cout << "aligned_:        " << (aligned_ ? "true" : "false") << "\n";
                std::cout << "pooled_height_:  " << pooled_height_ << "\n";
                std::cout << "pooled_width_:   " << pooled_width_ << "\n";
                std::cout << "levels_num:      " << levels_num << "\n";
                std::cout << "num_rois:        " << num_rois << "\n";
                std::cout << "channels_num:    " << channels_num << "\n";
                std::cout << "feaxels_per_roi: " << feaxels_per_roi << "\n\n";

                const float* input_rois = inputs[input_rois_port].data();

                std::vector<int64_t> level_ids(num_rois, 0);
                redistribute_rois(input_rois, level_ids.data(), num_rois, levels_num);
                std::cout << "level_ids:             ";
                for (auto x : level_ids)
                {
                    std::cout << x << ", ";
                }
                std::cout << "\n\n";

                std::vector<float> reordered_rois(4 * num_rois, 0);
                std::vector<int64_t> original_rois_mapping(num_rois, 0);
                reord(input_rois,
                      level_ids.data(),
                      num_rois,
                      4,
                      reordered_rois.data(),
                      original_rois_mapping.data());
                std::cout << "reordered_rois:        ";
                for (auto x : reordered_rois)
                {
                    std::cout << x << ", ";
                }
                std::cout << "\n\n";
                std::cout << "original_rois_mapping: ";
                for (auto x : original_rois_mapping)
                {
                    std::cout << x << ", ";
                }
                std::cout << "\n\n";

                std::vector<int64_t> rois_per_level;
                split_points(level_ids, rois_per_level, levels_num + 1);
                std::cout << "rois_per_level:        ";
                for (auto x : rois_per_level)
                {
                    std::cout << x << ", ";
                }
                std::cout << "\n\n";

                std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
                std::cout << "Loop through levels...\n";
                for (int64_t i = 0; i < levels_num; ++i)
                {
                    std::cout << "    i:                 " << i << "\n";
                    const int64_t level_rois_offset = rois_per_level[i];
                    const int64_t level_rois_num = rois_per_level[i + 1] - level_rois_offset;
                    std::cout << "    level_rois_offset: " << level_rois_offset << "\n";
                    std::cout << "    level_rois_num:    " << level_rois_num << "\n";
                    if (level_rois_num > 0)
                    {
                        const float* featuremap = inputs[input_features_start_port + i].data();
                        const int64_t featuremap_height =
                            static_cast<int64_t>(input_shapes[input_features_start_port + i][2]);
                        const int64_t featuremap_width =
                            static_cast<int64_t>(input_shapes[input_features_start_port + i][3]);
                        std::cout << "    featuremap_height: " << featuremap_height << "\n";
                        std::cout << "    featuremap_width:  " << featuremap_width << "\n";
                        ROIAlignForward_cpu_kernel<float>(
                            feaxels_per_roi * level_rois_num,
                            featuremap,
                            1.0f / pyramid_scales_[i],
                            channels_num,
                            featuremap_height,
                            featuremap_width,
                            pooled_height_,
                            pooled_width_,
                            sampling_ratio_,
                            &reordered_rois[4 * level_rois_offset],
                            aligned_,
                            &output_rois_features_temp[feaxels_per_roi * level_rois_offset]);
                    }
                }

                std::vector<int64_t> dummy_mapping(num_rois, 0);
                reord(output_rois_features_temp.data(),
                      original_rois_mapping.data(),
                      num_rois,
                      feaxels_per_roi,
                      output_rois_features,
                      dummy_mapping.data());

                memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
            }

            void experimental_detectron_roi_feature_extractor_postprocessing(
                void* prois_features,
                void* prois,
                const ngraph::element::Type output_type,
                const std::vector<float>& output_rois_features,
                const std::vector<float>& output_rois,
                const Shape& output_rois_features_shape,
                const Shape& output_rois_shape)
            {
                size_t output_rois_features_size = shape_size(output_rois_features_shape);
                size_t output_rois_size = shape_size(output_rois_shape);

                switch (output_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* output_rois_features_ptr =
                        reinterpret_cast<bfloat16*>(prois_features);
                    bfloat16* output_rois_ptr = reinterpret_cast<bfloat16*>(prois);
                    for (size_t i = 0; i < output_rois_features_size; ++i)
                    {
                        output_rois_features_ptr[i] = bfloat16(output_rois_features[i]);
                    }
                    for (size_t i = 0; i < output_rois_size; ++i)
                    {
                        output_rois_ptr[i] = bfloat16(output_rois[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* output_rois_features_ptr = reinterpret_cast<float16*>(prois_features);
                    float16* output_rois_ptr = reinterpret_cast<float16*>(prois);
                    for (size_t i = 0; i < output_rois_features_size; ++i)
                    {
                        output_rois_features_ptr[i] = float16(output_rois_features[i]);
                    }
                    for (size_t i = 0; i < output_rois_size; ++i)
                    {
                        output_rois_ptr[i] = float16(output_rois[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* output_rois_features_ptr = reinterpret_cast<float*>(prois_features);
                    float* output_rois_ptr = reinterpret_cast<float*>(prois);
                    memcpy(output_rois_features_ptr,
                           output_rois_features.data(),
                           output_rois_features_size * sizeof(float));
                    memcpy(output_rois_ptr, output_rois.data(), output_rois_size * sizeof(float));
                }
                break;
                default:;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
