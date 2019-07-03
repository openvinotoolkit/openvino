// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// There are some code snippets in this file.
// Original source file is avaialble here (Copyright (c) 2018 Facebook, MIT License):
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/cpu/ROIAlign_cpu.cpp
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

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
void pre_calc_for_bilinear_interpolate(
    const int height,
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
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

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

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
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
void ROIAlignForward_cpu_kernel(
    const int nthreads,
    const T* bottom_data,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
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

    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : static_cast<int>(ceil(roi_height / pooled_height));  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(roi_width / pooled_width));

    // We do average (integral) pooling inside a bin
    const T count = static_cast<T>(roi_bin_grid_h * roi_bin_grid_w);  // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
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
      const T* offset_bottom_data =
          bottom_data + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
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
      }  // for ph
    }  // for c
  });
}


void redistribute_rois(const float* rois, int* level_ids,
                       const int num_rois, const int levels_num) {
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
            target_level = std::max<int>(0, std::min<int>(levels_num - 1, target_level));
        }

        level_ids[i] = target_level;
    }
}


void reorder(const float* src_data, const int* ranks, const int n, const int step, float* dst_data,
             int* dst_mapping) {
    std::iota(dst_mapping, dst_mapping + n, 0);
    std::sort(dst_mapping, dst_mapping + n, [&ranks](size_t i1, size_t i2) {return ranks[i1] < ranks[i2];});
    for (int i = 0; i < n; ++i) {
        const int j = dst_mapping[i];
        assert(0 <= j && j < n);
        std::memcpy(dst_data + i * step, src_data + j * step, sizeof(float) * step);
    }
}

void split_points(const std::vector<int>& ids, std::vector<int>& rois_per_level, const int levels_num) {
    rois_per_level.clear();
    rois_per_level.resize(levels_num, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        assert(0 <= ids[i] && ids[i] < levels_num);
        rois_per_level[ids[i]]++;
    }
    for (int i = 1; i < levels_num; ++i) {
        rois_per_level[i] += rois_per_level[i - 1];
    }
    rois_per_level.insert(rois_per_level.begin(), 0);
}


void reorder_rois(const float *rois, const int* ids, int* mapping, const int rois_num,
                  float * reordered_rois, std::vector<int>& rois_per_level, const int levels_num) {
    rois_per_level.clear();
    rois_per_level.resize(levels_num, 0);
    for (int i = 0; i < rois_num; ++i) {
        assert(0 <= ids[i] && ids[i] < levels_num);
        rois_per_level[ids[i]]++;
    }
    for (int i = 1; i < levels_num; ++i) {
        rois_per_level[i] += rois_per_level[i - 1];
    }
    rois_per_level.insert(rois_per_level.begin(), 0);

    std::vector<int> level_counter = rois_per_level;

    for (int i = 0; i < rois_num; ++i) {
        const int level = ids[i];
        assert(level < levels_num);
        const int j = level_counter[level];
        assert(0 <= j && j < rois_num);
        reordered_rois[j * 4 + 0] = rois[i * 4 + 0];
        reordered_rois[j * 4 + 1] = rois[i * 4 + 1];
        reordered_rois[j * 4 + 2] = rois[i * 4 + 2];
        reordered_rois[j * 4 + 3] = rois[i * 4 + 3];
        level_counter[level]++;
    }
}

class ExperimentalDetectronROIFeatureExtractorImpl: public ExtLayerBase {
private:
    const int INPUT_ROIS {0};
    const int INPUT_FEATURES_START {1};

    const int OUTPUT_ROI_FEATURES {0};
    const int OUTPUT_ROIS {1};

public:
    explicit ExperimentalDetectronROIFeatureExtractorImpl(const CNNLayer* layer) {
        try {
            output_dim_ = layer->GetParamAsInt("output_size");
            pyramid_scales_ = layer->GetParamAsInts("pyramid_scales");
            sampling_ratio_ = layer->GetParamAsInt("sampling_ratio");
            pooled_height_ = output_dim_;
            pooled_width_ = output_dim_;

            std::vector<DataConfigurator> inputs_layouts(layer->insData.size(), DataConfigurator(ConfLayout::PLN));
            std::vector<DataConfigurator> outputs_layouts(layer->outData.size(), DataConfigurator(ConfLayout::PLN));
            addConfig(layer, inputs_layouts, outputs_layouts);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int levels_num = inputs.size() - INPUT_FEATURES_START;
        const int num_rois = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
        const int channels_num = inputs[INPUT_FEATURES_START]->getTensorDesc().getDims()[1];
        const int feaxels_per_roi = pooled_height_ * pooled_width_ * channels_num;

        auto *input_rois = inputs[INPUT_ROIS]->buffer().as<const float *>();
        auto *output_rois_features = outputs[OUTPUT_ROI_FEATURES]->buffer().as<float *>();
        float *output_rois = nullptr;
        if (OUTPUT_ROIS < static_cast<int>(outputs.size())) {
            output_rois = outputs[OUTPUT_ROIS]->buffer().as<float *>();
        }

        std::vector<int> level_ids(num_rois, 0);
        redistribute_rois(input_rois, reinterpret_cast<int *>(&level_ids[0]), num_rois, levels_num);

        std::vector<float> reordered_rois(4 * num_rois, 0);
        std::vector<int> original_rois_mapping(num_rois, 0);
        reorder(input_rois, &level_ids[0], num_rois, 4, &reordered_rois[0], &original_rois_mapping[0]);

        std::vector<int> rois_per_level;
        split_points(level_ids, rois_per_level, levels_num + 1);

        std::vector<float> output_rois_features_temp(feaxels_per_roi * num_rois, 0);
        for (int i = 0; i < levels_num; ++i) {
            const int level_rois_offset = rois_per_level[i];
            const int level_rois_num = rois_per_level[i + 1] - level_rois_offset;
            if (level_rois_num > 0) {
                auto *featuremap = inputs[INPUT_FEATURES_START + i]->buffer().as<const float *>();
                const int featuremap_height = inputs[INPUT_FEATURES_START + i]->getTensorDesc().getDims()[2];
                const int featuremap_width = inputs[INPUT_FEATURES_START + i]->getTensorDesc().getDims()[3];
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
                    &output_rois_features_temp[feaxels_per_roi * level_rois_offset]);
            }
        }

        std::vector<int> dummy_mapping(num_rois, 0);
        reorder(&output_rois_features_temp[0], &original_rois_mapping[0], num_rois, feaxels_per_roi,
                output_rois_features, &dummy_mapping[0]);
        if (output_rois != nullptr) {
            std::memcpy(output_rois, input_rois, 4 * num_rois * sizeof(float));
        }

        return OK;
    }

private:
    int output_dim_ = 0;
    int pooled_height_ = 0;
    int pooled_width_ = 0;
    std::vector<int> pyramid_scales_;
    int sampling_ratio_ = 0;

    int channels = 0;
    int height = 0;
    int width = 0;

    int nn = 0;
    int nc = 0;
    int nh = 0;
    int nw = 0;
};

REG_FACTORY_FOR(ImplFactory<ExperimentalDetectronROIFeatureExtractorImpl>, ExperimentalDetectronROIFeatureExtractor);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
