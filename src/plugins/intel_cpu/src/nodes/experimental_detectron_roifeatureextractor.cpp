// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_roifeatureextractor.h"

#include <algorithm>
#include <cmath>
#include <openvino/opsets/opset6.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu::node {
namespace {

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(const int height,
                                       const int width,
                                       const int pooled_height,
                                       const int pooled_width,
                                       const int iy_upper,
                                       const int ix_upper,
                                       T roi_start_h,
                                       T roi_start_w,
                                       T bin_size_h,
                                       T bin_size_w,
                                       int roi_bin_grid_h,
                                       int roi_bin_grid_w,
                                       std::vector<PreCalc<T>>& pre_calc) {
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
            for (int iy = 0; iy < iy_upper; iy++) {
                const T yy = roi_start_h + ph * bin_size_h +
                             static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++) {
                    const T xx = roi_start_w + pw * bin_size_w +
                                 static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
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

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    auto y_low = static_cast<int>(y);
                    auto x_low = static_cast<int>(x);
                    int y_high = 0;
                    int x_high = 0;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y = (T)y_low;
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x = (T)x_low;
                    } else {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = static_cast<T>(1) - ly, hx = static_cast<T>(1) - lx;
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                    // save weights and indices
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
void ROIAlignForward_cpu_kernel(const int nthreads,
                                const T* bottom_data,
                                const T& spatial_scale,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio,
                                const T* bottom_rois,
                                const bool aligned,
                                T* top_data) {
    int roi_cols = 4;

    int n_rois = nthreads / channels / pooled_width / pooled_height;
    // (n, c, ph, pw) is an element in the pooled output
    parallel_for(n_rois, [&](size_t n) {
        int index_n = n * channels * pooled_width * pooled_height;

        // roi could have 4 or 5 columns
        const T* offset_bottom_rois = bottom_rois + n * roi_cols;
        int roi_batch_ind = 0;
        if (roi_cols == 5) {
            roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
            offset_bottom_rois++;
        }

        T offset = aligned ? (T)0.5 : (T)0.0;
        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[0] * spatial_scale - offset;
        T roi_start_h = offset_bottom_rois[1] * spatial_scale - offset;
        T roi_end_w = offset_bottom_rois[2] * spatial_scale - offset;
        T roi_end_h = offset_bottom_rois[3] * spatial_scale - offset;

        // Force malformed ROIs to be 1x1
        T roi_width = (std::max)(roi_end_w - roi_start_w, (T)1.);
        T roi_height = (std::max)(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
                                 ? sampling_ratio
                                 : static_cast<int>(std::ceil(roi_height / pooled_height));  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(std::ceil(roi_width / pooled_width));

        // We do average (integral) pooling inside a bin
        const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);  // e.g. = 4

        // we want to precalculate indices and weights shared by all chanels,
        // this is the key point of optimiation
        std::vector<PreCalc<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
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

        for (int c = 0; c < channels; c++) {
            int index_n_c = index_n + c * pooled_width * pooled_height;
            const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++) {
                for (int pw = 0; pw < pooled_width; pw++) {
                    int index = index_n_c + ph * pooled_width + pw;

                    T output_val = 0.;
                    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
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
    });
}

void redistribute_rois(const float* rois, int* level_ids, const int num_rois, const int levels_num) {
    const float canonical_scale = 224.0f;
    const int canonical_level = 2;

    for (int i = 0; i < num_rois; ++i) {
        const float x0 = rois[4 * i + 0];
        const float y0 = rois[4 * i + 1];
        const float x1 = rois[4 * i + 2];
        const float y1 = rois[4 * i + 3];

        int target_level = levels_num;
        float area = (x1 - x0) * (y1 - y0);
        if (area > 0) {
            area = std::sqrt(area) / canonical_scale;
            area = std::log2(area + 1e-6f);
            target_level = static_cast<int>(std::floor(area + canonical_level));
            target_level = (std::max)(0, (std::min)(levels_num - 1, target_level));
        }

        level_ids[i] = target_level;
    }
}

void reord(const float* src_data, const int* ranks, const int n, const int step, float* dst_data, int* dst_mapping) {
    std::iota(dst_mapping, dst_mapping + n, 0);
    std::sort(dst_mapping, dst_mapping + n, [&ranks](size_t i1, size_t i2) {
        return ranks[i1] < ranks[i2];
    });
    for (int i = 0; i < n; ++i) {
        const int j = dst_mapping[i];
        assert(0 <= j && j < n);
        cpu_memcpy(dst_data + i * step, src_data + j * step, sizeof(float) * step);
    }
}

void split_points(const std::vector<int>& ids, std::vector<int>& rois_per_level, const int levels_num) {
    rois_per_level.clear();
    rois_per_level.resize(levels_num, 0);
    for (int id : ids) {
        rois_per_level[id]++;
    }
    for (int i = 1; i < levels_num; ++i) {
        rois_per_level[i] += rois_per_level[i - 1];
    }
    rois_per_level.insert(rois_per_level.begin(), 0);
}

}  // namespace

bool ExperimentalDetectronROIFeatureExtractor::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                                                    std::string& errorMessage) noexcept {
    try {
        const auto roiFeatureExtractor =
            ov::as_type_ptr<const ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op);
        if (!roiFeatureExtractor) {
            errorMessage = "Only opset6 ExperimentalDetectronROIFeatureExtractor operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(const std::shared_ptr<ov::Node>& op,
                                                                                   const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto roiFeatureExtractor = ov::as_type_ptr<const ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op);
    const auto& attr = roiFeatureExtractor->get_attrs();
    output_dim_ = attr.output_size;
    pyramid_scales_ = attr.pyramid_scales;
    sampling_ratio_ = attr.sampling_ratio;
    aligned_ = attr.aligned;
    pooled_height_ = output_dim_;
    pooled_width_ = output_dim_;
}

void ExperimentalDetectronROIFeatureExtractor::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
    }

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronROIFeatureExtractor::execute(const dnnl::stream& strm) {
    const int levels_num = inputShapes.size() - INPUT_FEATURES_START;
    const int num_rois = getParentEdgeAt(INPUT_ROIS)->getMemory().getStaticDims()[0];
    const int channels_num = getParentEdgeAt(INPUT_FEATURES_START)->getMemory().getStaticDims()[1];
    const int feaxels_per_roi = pooled_height_ * pooled_width_ * channels_num;

    auto* input_rois = getSrcDataAtPortAs<const float>(INPUT_ROIS);
    auto* output_rois_features = getDstDataAtPortAs<float>(OUTPUT_ROI_FEATURES);
    float* output_rois = nullptr;
    if (OUTPUT_ROIS < outputShapes.size()) {
        output_rois = getDstDataAtPortAs<float>(OUTPUT_ROIS);
    }

    std::vector<int> level_ids(num_rois, 0);
    redistribute_rois(input_rois, reinterpret_cast<int*>(&level_ids[0]), num_rois, levels_num);

    std::vector<float> reordered_rois(4 * num_rois, 0);
    std::vector<int> original_rois_mapping(num_rois, 0);
    reord(input_rois, &level_ids[0], num_rois, 4, &reordered_rois[0], &original_rois_mapping[0]);

    std::vector<int> rois_per_level;
    split_points(level_ids, rois_per_level, levels_num + 1);

    std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
    for (int i = 0; i < levels_num; ++i) {
        const int level_rois_offset = rois_per_level[i];
        const int level_rois_num = rois_per_level[i + 1] - level_rois_offset;
        if (level_rois_num > 0) {
            auto* featuremap = getSrcDataAtPortAs<const float>(INPUT_FEATURES_START + i);
            const int featuremap_height = getParentEdgeAt(INPUT_FEATURES_START + i)->getMemory().getStaticDims()[2];
            const int featuremap_width = getParentEdgeAt(INPUT_FEATURES_START + i)->getMemory().getStaticDims()[3];
            ROIAlignForward_cpu_kernel<float>(feaxels_per_roi * level_rois_num,
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

    std::vector<int> dummy_mapping(num_rois, 0);
    reord(&output_rois_features_temp[0],
          &original_rois_mapping[0],
          num_rois,
          feaxels_per_roi,
          output_rois_features,
          &dummy_mapping[0]);
    if (output_rois != nullptr) {
        cpu_memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
    }
}

bool ExperimentalDetectronROIFeatureExtractor::created() const {
    return getType() == Type::ExperimentalDetectronROIFeatureExtractor;
}

}  // namespace ov::intel_cpu::node
