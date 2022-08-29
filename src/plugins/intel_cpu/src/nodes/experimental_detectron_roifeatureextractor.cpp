// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <algorithm>

#include <ngraph/opsets/opset6.hpp>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "experimental_detectron_roifeatureextractor.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct LevelParams {
    const float *src;
    int height;
    int width;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
        const int n_roi,
        const int height,
        const int width,
        const int pooled_h,
        const int pooled_w,
        const int sampling_ratio_h,
        const int sampling_ratio_w,
        T y1,
        T x1,
        T bin_height,
        T bin_width,
        T sample_distance_h,
        T sample_distance_w,
        std::vector<std::vector<T>>& weights,
        std::vector<std::vector<int>>& indices) {
    constexpr size_t bli_params_num = 4u;
    int pre_calc_index = 0;

    weights[n_roi].resize(bli_params_num * sampling_ratio_h * sampling_ratio_w * pooled_h * pooled_w);
    indices[n_roi].resize(bli_params_num * sampling_ratio_h * sampling_ratio_w * pooled_h * pooled_w);

    for (int ph = 0; ph < pooled_h; ph++) {
        for (int pw = 0; pw < pooled_w; pw++) {
            for (int iy = 0; iy < sampling_ratio_h; iy++) {
                const T yy = y1 + ph * bin_height + static_cast<T>(iy + .5f) * sample_distance_h;  // e.g., 0.5, 1.5
                for (int ix = 0; ix < sampling_ratio_w; ix++) {
                    const T xx = x1 + pw * bin_width + static_cast<T>(ix + .5f) * sample_distance_w;

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        // weights and indices are 0
                        weights[n_roi][pre_calc_index] = static_cast<T>(0);
                        weights[n_roi][pre_calc_index + 1] = static_cast<T>(0);
                        weights[n_roi][pre_calc_index + 2] = static_cast<T>(0);
                        weights[n_roi][pre_calc_index + 3] = static_cast<T>(0);

                        indices[n_roi][pre_calc_index] = 0;
                        indices[n_roi][pre_calc_index + 1] = 0;
                        indices[n_roi][pre_calc_index + 1] = 0;
                        indices[n_roi][pre_calc_index + 1] = 0;

                        pre_calc_index += bli_params_num;
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    int y_low = static_cast<int>(y);
                    int x_low = static_cast<int>(x);
                    int y_high = 0;
                    int x_high = 0;

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
                    T hy = static_cast<T>(1) - ly, hx = static_cast<T>(1) - lx;

                    // save weights and indices
                    weights[n_roi][pre_calc_index] = hy * hx;
                    weights[n_roi][pre_calc_index + 1] = hy * lx;
                    weights[n_roi][pre_calc_index + 2] = ly * hx;
                    weights[n_roi][pre_calc_index + 3] = ly * lx;

                    indices[n_roi][pre_calc_index] = y_low * width + x_low;
                    indices[n_roi][pre_calc_index + 1] = y_low * width + x_high;
                    indices[n_roi][pre_calc_index + 2] = y_high * width + x_low;
                    indices[n_roi][pre_calc_index + 3] = y_high * width + x_high;

                    pre_calc_index += bli_params_num;
                }
            }
        }
    }
}

void roiTargetLevels(const float* rois, int* levels, const size_t num_rois, const int num_levels) {
    constexpr float canonical_scale = 224.0f;
    constexpr int canonical_level = 2;

    for (size_t i = 0; i < num_rois; ++i) {
        const float x0 = rois[4 * i + 0];
        const float y0 = rois[4 * i + 1];
        const float x1 = rois[4 * i + 2];
        const float y1 = rois[4 * i + 3];

        int target_level = num_levels;
        float area = (x1 - x0) * (y1 - y0);
        if (area > 0) {
            area = std::sqrt(area) / canonical_scale;
            area = std::log2(area + 1e-6f);
            target_level = static_cast<int>(std::floor(area + canonical_level));
            target_level = std::max(0, std::min(num_levels - 1, target_level));
        }

        levels[i] = target_level;
    }
}

} // namespace

bool ExperimentalDetectronROIFeatureExtractor::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
                                                                              std::string& errorMessage) noexcept {
    try {
        const auto roiFeatureExtractor = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronROIFeatureExtractor>(op);
        if (!roiFeatureExtractor) {
            errorMessage = "Only opset6 ExperimentalDetectronROIFeatureExtractor operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor
        (const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
                WeightsSharing::Ptr &cache) : Node(op, eng, cache, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto roiFeatureExtractor = std::dynamic_pointer_cast<const ngraph::opset6::ExperimentalDetectronROIFeatureExtractor>(op);
    const auto &attr = roiFeatureExtractor->get_attrs();
    output_dim_ = attr.output_size;
    pyramid_scales_ = attr.pyramid_scales;
    sampling_ratio_ = attr.sampling_ratio;
    aligned_ = attr.aligned;
    pooled_height_ = output_dim_;
    pooled_width_ = output_dim_;
}

void ExperimentalDetectronROIFeatureExtractor::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (int i = 0; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void ExperimentalDetectronROIFeatureExtractor::createPrimitive() {
    Node::createPrimitive();

    const bool jit_is_beneficial = (sampling_ratio_ >= 3 || sampling_ratio_ == 0);
    if (!roi_align_kernel_ && jit_is_beneficial) {
        createJitKernel(getParentEdgeAt(0)->getMemoryPtr()->getDesc().getPrecision(), ROIAlignLayoutType::ncsp); 
    }
}

void ExperimentalDetectronROIFeatureExtractor::createJitKernel(const InferenceEngine::Precision& dataPrec, const ROIAlignLayoutType& selectLayout) {
    auto jcp = jit_roi_align_params();
    jcp.alg = Algorithm::ROIAlignAvg;
    jcp.data_prc = dataPrec;
    jcp.data_size = dataPrec.size();
    jcp.layout = selectLayout;
    jcp.pooled_h = pooled_height_;
    jcp.pooled_w = pooled_width_;

    if (mayiuse(cpu::x64::avx512_core)) {
        roi_align_kernel_.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx512_core>(jcp));
    } else if (mayiuse(cpu::x64::avx2)) {
        roi_align_kernel_.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::avx2>(jcp));
    } else if (mayiuse(cpu::x64::sse41)) {
        roi_align_kernel_.reset(new jit_uni_roi_align_kernel_f32<cpu::x64::sse41>(jcp));
    }

    if (roi_align_kernel_)
        roi_align_kernel_->create_ker();
}

void ExperimentalDetectronROIFeatureExtractor::execute(dnnl::stream strm) {
    const int levels_num = inputShapes.size() - INPUT_FEATURES_START;
    const int num_rois = getParentEdgeAt(INPUT_ROIS)->getMemory().getStaticDims()[0];
    const int channels_num = getParentEdgeAt(INPUT_FEATURES_START)->getMemory().getStaticDims()[1];

    auto *input_rois = reinterpret_cast<const float *>(getParentEdgeAt(INPUT_ROIS)->getMemoryPtr()->GetPtr());
    auto *output_rois_features = reinterpret_cast<float *>(getChildEdgesAtPort(OUTPUT_ROI_FEATURES)[0]->getMemoryPtr()->GetPtr());
    float *output_rois = nullptr;
    if (OUTPUT_ROIS < outputShapes.size()) {
        output_rois = reinterpret_cast<float *>(getChildEdgesAtPort(OUTPUT_ROIS)[0]->getMemoryPtr()->GetPtr());
    }

    std::vector<std::vector<float>> weights(num_rois);
    std::vector<std::vector<int>> indices(num_rois);
    std::vector<int> levels(num_rois);
    std::vector<LevelParams> level_params(levels_num);

    for (size_t i = 0; i < levels_num; ++i) {
        level_params[i].src = reinterpret_cast<const float *>(getParentEdgeAt(INPUT_FEATURES_START + i)->getMemoryPtr()->GetPtr());
        level_params[i].height = getParentEdgeAt(INPUT_FEATURES_START + i)->getMemory().getStaticDims()[2];
        level_params[i].width = getParentEdgeAt(INPUT_FEATURES_START + i)->getMemory().getStaticDims()[3];
    }

    roiTargetLevels(input_rois, &levels[0], num_rois, levels_num);

    const float offset = aligned_ ? 0.5f : 0.0f;

    parallel_for(num_rois, [&](int n) {
        const int level = levels[n];
        if (level >= levels_num) {
            const size_t output_feature_size = channels_num * pooled_height_ * pooled_width_;
            memset(&output_rois_features[n * output_feature_size], 0, output_feature_size * sizeof(float));
            return;
        }

        // precalc

        const float spatial_scale = 1.0f / pyramid_scales_[level];

        const float x1 = input_rois[4 * n + 0] * spatial_scale - offset;
        const float y1 = input_rois[4 * n + 1] * spatial_scale - offset;
        const float x2 = input_rois[4 * n + 2] * spatial_scale - offset;
        const float y2 = input_rois[4 * n + 3] * spatial_scale - offset;

        // force malformed ROIs to be 1x1
        const float roi_width = std::max(x2 - x1, 1.f);
        const float roi_height = std::max(y2 - y1, 1.f);

        const float bin_height = roi_height / pooled_height_;
        const float bin_width = roi_width / pooled_width_;

        const int sampling_ratio_h = sampling_ratio_ == 0 ? static_cast<int>(ceil(bin_height)) : sampling_ratio_;
        const int sampling_ratio_w = sampling_ratio_ == 0 ? static_cast<int>(ceil(bin_width)) : sampling_ratio_;
        const int samples_per_bin = sampling_ratio_h * sampling_ratio_w;

        const float sample_distance_h = bin_height / sampling_ratio_h;
        const float sample_distance_w = bin_width / sampling_ratio_w;

        pre_calc_for_bilinear_interpolate(
            n,
            level_params[level].height,
            level_params[level].width,
            pooled_height_,
            pooled_width_,
            sampling_ratio_h,
            sampling_ratio_w,
            y1,
            x1,
            bin_height,
            bin_width,
            sample_distance_h,
            sample_distance_w,
            weights,
            indices);

        // pooling

        const float *src_data = level_params[level].src;
        const int height = level_params[level].height;
        const int width = level_params[level].width;

        const float samples_per_bin_invert = 1.f / samples_per_bin;

        size_t channel_src_offset = 0;
        const size_t channel_src_offset_inc = height * width;
        size_t bin_dst_offset = n * channels_num * pooled_height_ * pooled_width_;

        if (roi_align_kernel_) {
            auto arg = jit_roi_align_call_args();
            arg.scale = static_cast<const float*>(&samples_per_bin_invert);
            arg.num_samples = samples_per_bin;

            for (size_t i = 0; i < channels_num; ++i) {
                arg.src = static_cast<const void*>(&src_data[channel_src_offset]);
                channel_src_offset += channel_src_offset_inc;

                size_t param_offset = 0;
                const size_t param_offset_inc = 4 * samples_per_bin;
                for (size_t ph = 0; ph < pooled_height_; ++ph) {
                    for (size_t pw = 0; pw < pooled_width_; ++pw) {
                        arg.buffer = static_cast<void*>(&indices[n][param_offset]);
                        arg.weights = static_cast<const float*>(&weights[n][param_offset]);
                        param_offset += param_offset_inc;

                        arg.dst = static_cast<void*>(&output_rois_features[bin_dst_offset]);
                        ++bin_dst_offset;

                        (*roi_align_kernel_)(&arg);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < channels_num; ++i) {
                size_t param_offset = 0;
                for (size_t ph = 0; ph < pooled_height_; ++ph) {
                    for (size_t pw = 0; pw < pooled_width_; ++pw) {
                        float pooled_value = 0.f;
                        for (size_t ind = 0; ind < samples_per_bin; ++ind) {
                            const float src0 = src_data[channel_src_offset + indices[n][param_offset]];
                            const float src1 = src_data[channel_src_offset + indices[n][param_offset + 1]];
                            const float src2 = src_data[channel_src_offset + indices[n][param_offset + 2]];
                            const float src3 = src_data[channel_src_offset + indices[n][param_offset + 3]];

                            float sample_value =
                                weights[n][param_offset] * src0 +
                                weights[n][param_offset + 1] * src1 +
                                weights[n][param_offset + 2] * src2 +
                                weights[n][param_offset + 3] * src3;

                            param_offset += 4;

                            pooled_value += sample_value * samples_per_bin_invert;
                        }

                        output_rois_features[bin_dst_offset] = pooled_value;
                        ++bin_dst_offset;
                    }
                }

                channel_src_offset += channel_src_offset_inc;
            }
        }
    });

    if (output_rois != nullptr) {
        cpu_memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
    }
}

bool ExperimentalDetectronROIFeatureExtractor::created() const {
    return getType() == Type::ExperimentalDetectronROIFeatureExtractor;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
