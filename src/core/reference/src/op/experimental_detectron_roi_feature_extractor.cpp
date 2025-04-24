// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/experimental_detectron_roi_feature_extractor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

#include "openvino/core/shape.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#    if defined(__linux__) && defined(OPENVINO_ARCH_X86) && \
        (__GNUC__ == 7 && __GNUC_MINOR__ == 5 && __GNUC_PATCHLEVEL__ == 0)
#        define _OV_DISABLE_GCC_OPTIMIZATION 1
#    else
#        define _OV_DISABLE_GCC_OPTIMIZATION 0
#    endif
#else
#    define _OV_DISABLE_GCC_OPTIMIZATION 0
#endif

namespace {
constexpr int64_t input_rois_port = 0;
constexpr int64_t input_features_start_port = 1;

void redistribute_rois(const std::vector<float>& rois, std::vector<int64_t>& level_ids, const int64_t levels_num) {
    const float canonical_scale = 224.0f;
    const int64_t canonical_level = 2;
    const size_t num_rois = level_ids.size();

    for (size_t i = 0; i < num_rois; ++i) {
        const float x0 = rois[4 * i + 0];
        const float y0 = rois[4 * i + 1];
        const float x1 = rois[4 * i + 2];
        const float y1 = rois[4 * i + 3];

        int64_t target_level = levels_num;
        float area = (x1 - x0) * (y1 - y0);
        if (area > 0) {
            area = std::sqrt(area) / canonical_scale;
            area = std::log2(area + 1e-6f);
            target_level = static_cast<int64_t>(std::floor(area + canonical_level));
            target_level = std::max(static_cast<int64_t>(0), std::min(levels_num - 1, target_level));
        }

        level_ids[i] = target_level;
    }
}

void reord(const std::vector<float>& src_data,
           const std::vector<int64_t>& ranks,
           const int64_t step,
           float* dst_data,
           std::vector<int64_t>& dst_mapping) {
    int64_t n = static_cast<int64_t>(ranks.size());

    std::iota(dst_mapping.begin(), dst_mapping.end(), 0);
    std::sort(dst_mapping.begin(), dst_mapping.end(), [&ranks](int64_t i1, int64_t i2) {
        return ranks[i1] < ranks[i2];
    });
    for (int64_t i = 0; i < n; ++i) {
        const int64_t j = dst_mapping[i];
        memcpy(dst_data + i * step, src_data.data() + j * step, sizeof(float) * step);
    }
}

void split_points(const std::vector<int64_t>& ids, std::vector<int64_t>& rois_per_level, const int64_t levels_num) {
    rois_per_level.clear();
    rois_per_level.resize(levels_num, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        rois_per_level[ids[i]]++;
    }
    for (int64_t i = 1; i < levels_num; ++i) {
        rois_per_level[i] += rois_per_level[i - 1];
    }
    rois_per_level.insert(rois_per_level.begin(), 0);
}

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
    int64_t pos1;
    int64_t pos2;
    int64_t pos3;
    int64_t pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

// The function pre_calc_for_bilinear_interpolate() gives incorrect results for -O3 optimization level, when IE
// is compiled using GCC 7.5.0 on Ubuntu 18.04 32-bit. But results are correct, for example, if we use Clang 10.0
// on Ubuntu 18.04 32-bit with -O3 optimization level. Next, the function pre_calc_for_bilinear_interpolate()
// gives incorrect results after compiling by GCC 7.5.0 or Clang 10 in Ubuntu 18.04 32-bit, if the optimization
// level is -O1 or -O2. Finally, the function gives correct result in Ubuntu 18.04 32-bit, if the optimization
// level is -O0.
#if _OV_DISABLE_GCC_OPTIMIZATION
#    pragma GCC push_options
#    pragma GCC optimize("-O0")
#endif
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
                                       std::vector<PreCalc<T>>& pre_calc) {
    int64_t pre_calc_index = 0;
    for (int64_t ph = 0; ph < pooled_height; ph++) {
        for (int64_t pw = 0; pw < pooled_width; pw++) {
            for (int64_t iy = 0; iy < iy_upper; iy++) {
                for (int64_t ix = 0; ix < ix_upper; ix++) {
                    T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
                          (static_cast<T>(iy) + static_cast<T>(0.5f)) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                    T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
                          (static_cast<T>(ix) + static_cast<T>(0.5f)) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    // deal with: inverse elements are out of feature map boundary
                    if (y < static_cast<T>(-1.0f) || y > static_cast<T>(height) || x < static_cast<T>(-1.0f) ||
                        x > static_cast<T>(width)) {
                        // empty
                        pre_calc_index += 1;
                        continue;
                    }

                    y = std::max(y, static_cast<T>(0.0f));
                    x = std::max(x, static_cast<T>(0.0f));

                    int64_t y_low = static_cast<int64_t>(y);
                    int64_t x_low = static_cast<int64_t>(x);
                    int64_t y_high = 0;
                    int64_t x_high = 0;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y = static_cast<T>(y_low);
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x = static_cast<T>(x_low);
                    } else {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = static_cast<T>(1.0) - ly;
                    T hx = static_cast<T>(1.0) - lx;

                    // save weights and indeces
                    PreCalc<T> pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = hy * hx;
                    pc.w2 = hy * lx;
                    pc.w3 = ly * hx;
                    pc.w4 = ly * lx;
                    pre_calc.at(pre_calc_index) = pc;

                    pre_calc_index += 1;
                }
            }
        }
    }
}
#if _OV_DISABLE_GCC_OPTIMIZATION
#    pragma GCC pop_options
#endif

template <typename T>
void ROIAlignForward(const int64_t nthreads,
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
                     T* top_data) {
    int64_t roi_cols = 4;

    int64_t n_rois = nthreads / channels / pooled_width / pooled_height;
    // (n, c, ph, pw) is an element in the pooled output
    for (int64_t n = 0; n < n_rois; ++n) {
        int64_t index_n = n * channels * pooled_width * pooled_height;

        // roi could have 4 or 5 columns
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int64_t roi_batch_ind = 0;
        if (roi_cols == 5) {
            roi_batch_ind = static_cast<int64_t>(offset_bottom_rois[0]);
            offset_bottom_rois++;
        }

        T offset = aligned ? static_cast<T>(0.5) : static_cast<T>(0.0);
        // Do not use rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[0] * spatial_scale - offset;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale - offset;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale - offset;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale - offset;

        // Force malformed ROIs to be 1x1
        T roi_width = std::max(roi_end_w - roi_start_w, static_cast<T>(1.0));
        T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(1.0));
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int64_t roi_bin_grid_h =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_height / pooled_height));
        int64_t roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);

        // we want to precalculate indices and weights shared by all channels,
        // this is the key point of optimization
        std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);

        pre_calc_for_bilinear_interpolate<T>(height,
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

        for (int64_t c = 0; c < channels; c++) {
            int64_t index_n_c = index_n + c * pooled_width * pooled_height;
            const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
            int64_t pre_calc_index = 0;

            for (int64_t ph = 0; ph < pooled_height; ph++) {
                for (int64_t pw = 0; pw < pooled_width; pw++) {
                    int64_t index = index_n_c + ph * pooled_width + pw;
                    T output_val = 0.;
                    for (int64_t iy = 0; iy < roi_bin_grid_h; iy++) {
                        for (int64_t ix = 0; ix < roi_bin_grid_w; ix++) {
                            PreCalc<T> pc = pre_calc[pre_calc_index];
                            output_val += pc.w1 * offset_bottom_data[pc.pos1] + pc.w2 * offset_bottom_data[pc.pos2] +
                                          pc.w3 * offset_bottom_data[pc.pos3] + pc.w4 * offset_bottom_data[pc.pos4];

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
}  // namespace

namespace ov {
namespace reference {
void experimental_detectron_roi_feature_extractor(
    const std::vector<std::vector<float>>& inputs,
    const std::vector<Shape>& input_shapes,
    const op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes& attrs,
    float* output_rois_features,
    float* output_rois) {
    int64_t output_dim = attrs.output_size;
    auto pyramid_scales = attrs.pyramid_scales;
    int64_t sampling_ratio = attrs.sampling_ratio;
    bool aligned = attrs.aligned;
    int64_t pooled_height = output_dim;
    int64_t pooled_width = output_dim;

    const int64_t levels_num = static_cast<int64_t>(inputs.size() - input_features_start_port);
    const int64_t num_rois = static_cast<int64_t>(input_shapes[input_rois_port][0]);
    const int64_t channels_num = static_cast<int64_t>(input_shapes[input_features_start_port][1]);
    const int64_t feaxels_per_roi = pooled_height * pooled_width * channels_num;

    const float* input_rois = inputs[input_rois_port].data();

    std::vector<int64_t> level_ids(num_rois, 0);
    redistribute_rois(inputs[input_rois_port], level_ids, levels_num);

    std::vector<float> reordered_rois(4 * num_rois, 0);
    std::vector<int64_t> original_rois_mapping(num_rois, 0);
    reord(inputs[input_rois_port], level_ids, 4, reordered_rois.data(), original_rois_mapping);

    std::vector<int64_t> rois_per_level;
    split_points(level_ids, rois_per_level, levels_num + 1);

    std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
    for (int64_t i = 0; i < levels_num; ++i) {
        const int64_t level_rois_offset = rois_per_level[i];
        const int64_t level_rois_num = rois_per_level[i + 1] - level_rois_offset;
        if (level_rois_num > 0) {
            const float* featuremap = inputs[input_features_start_port + i].data();
            const int64_t featuremap_height = static_cast<int64_t>(input_shapes[input_features_start_port + i][2]);
            const int64_t featuremap_width = static_cast<int64_t>(input_shapes[input_features_start_port + i][3]);
            ROIAlignForward<float>(feaxels_per_roi * level_rois_num,
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

    std::vector<int64_t> dummy_mapping(num_rois, 0);
    reord(output_rois_features_temp, original_rois_mapping, feaxels_per_roi, output_rois_features, dummy_mapping);

    memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
}

void experimental_detectron_roi_feature_extractor_postprocessing(void* prois_features,
                                                                 void* prois,
                                                                 const element::Type output_type,
                                                                 const std::vector<float>& output_rois_features,
                                                                 const std::vector<float>& output_rois,
                                                                 const Shape& output_rois_features_shape,
                                                                 const Shape& output_rois_shape) {
    size_t output_rois_features_size = shape_size(output_rois_features_shape);
    size_t output_rois_size = shape_size(output_rois_shape);

    switch (output_type) {
    case element::Type_t::bf16: {
        bfloat16* output_rois_features_ptr = reinterpret_cast<bfloat16*>(prois_features);
        bfloat16* output_rois_ptr = reinterpret_cast<bfloat16*>(prois);
        for (size_t i = 0; i < output_rois_features_size; ++i) {
            output_rois_features_ptr[i] = bfloat16(output_rois_features[i]);
        }
        for (size_t i = 0; i < output_rois_size; ++i) {
            output_rois_ptr[i] = bfloat16(output_rois[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* output_rois_features_ptr = reinterpret_cast<float16*>(prois_features);
        float16* output_rois_ptr = reinterpret_cast<float16*>(prois);
        for (size_t i = 0; i < output_rois_features_size; ++i) {
            output_rois_features_ptr[i] = float16(output_rois_features[i]);
        }
        for (size_t i = 0; i < output_rois_size; ++i) {
            output_rois_ptr[i] = float16(output_rois[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* output_rois_features_ptr = reinterpret_cast<float*>(prois_features);
        float* output_rois_ptr = reinterpret_cast<float*>(prois);
        memcpy(output_rois_features_ptr, output_rois_features.data(), output_rois_features_size * sizeof(float));
        memcpy(output_rois_ptr, output_rois.data(), output_rois_size * sizeof(float));
    } break;
    default:;
    }
}
}  // namespace reference
}  // namespace ov
