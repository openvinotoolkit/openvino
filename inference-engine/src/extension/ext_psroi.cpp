// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PSROIPoolingImpl: public ExtLayerBase {
public:
    explicit PSROIPoolingImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";
            // LayerSetUp
            output_dim_ = static_cast<size_t>(layer->GetParamAsInt("output_dim"));
            group_size_ = static_cast<size_t>(layer->GetParamAsInt("group_size"));
            spatial_scale_ = layer->GetParamAsFloat("spatial_scale");
            pooled_height_ = group_size_;
            pooled_width_ = group_size_;
            spatial_bins_x_ = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_x", 1));
            spatial_bins_y_ = static_cast<size_t>(layer->GetParamAsInt("spatial_bins_y", 1));
            mode_ = layer->GetParamAsString("mode", "average");

            SizeVector inDims = layer->insData[0].lock()->getTensorDesc().getDims();
            channels = static_cast<int>(inDims[1]);
            height = static_cast<int>(inDims[2]);
            width = static_cast<int>(inDims[3]);

            SizeVector outDims = layer->outData[0]->getTensorDesc().getDims();
            nn = static_cast<int>(outDims[0]);
            nc = static_cast<int>(outDims[1]);
            nh = static_cast<int>(outDims[2]);
            nw = static_cast<int>(outDims[3]);

            addConfig(layer, {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* dst_data = outputs[0]->buffer();
        const float *bottom_data_beginning = inputs[0]->buffer();
        const float *bottom_rois_beginning = inputs[1]->buffer();

        int real_rois = 0;
        for (; real_rois < nn; real_rois++) {
            const float *bottom_rois = bottom_rois_beginning + real_rois * 5;
            int roi_batch_ind = static_cast<int>(bottom_rois[0]);
            if (roi_batch_ind == -1) {
                break;
            }
        }

        size_t num_bins = spatial_bins_x_*spatial_bins_y_;

        parallel_for(real_rois, [&](int n) {
            const float* bottom_rois = bottom_rois_beginning + n * 5;
            int roi_batch_ind = static_cast<int>(bottom_rois[0]);
            float roi_start_w = 0.0f;
            float roi_start_h = 0.0f;
            float roi_end_w   = 0.0f;
            float roi_end_h   = 0.0f;
            float roi_width   = 0.0f;
            float roi_height  = 0.0f;

            if (mode_ == "bilinear") {
                roi_start_w = bottom_rois[1] * spatial_scale_;
                roi_start_h = bottom_rois[2] * spatial_scale_;
                roi_end_w = bottom_rois[3] * spatial_scale_;
                roi_end_h = bottom_rois[4] * spatial_scale_;
                roi_width  = roi_end_w - roi_start_w;
                roi_height = roi_end_h - roi_start_h;
            } else if (mode_ == "average") {
                roi_start_w = static_cast<float>(round(bottom_rois[1])) * spatial_scale_;
                roi_start_h = static_cast<float>(round(bottom_rois[2])) * spatial_scale_;
                roi_end_w   = static_cast<float>(round(bottom_rois[3]) + 1.0f) * spatial_scale_;
                roi_end_h   = static_cast<float>(round(bottom_rois[4]) + 1.0f) * spatial_scale_;
                // Force too small ROIs to be 1x1
                roi_width  = std::max<float>(roi_end_w - roi_start_w, 0.1f);  // avoid 0
                roi_height = std::max<float>(roi_end_h - roi_start_h, 0.1f);
            }

            for (int c = 0; c < nc; c++) {
                for (int h = 0; h < nh; h++) {
                    for (int w = 0; w < nw; w++) {
                        size_t index = n*nc*nh*nw + c*nh*nw + h*nw + w;
                        dst_data[index] = 0.0f;

                        if (mode_ == "average") {
                            float bin_size_h = roi_height / static_cast<float>(pooled_height_);
                            float bin_size_w = roi_width  / static_cast<float>(pooled_width_);

                            int hstart = static_cast<int>(floor(static_cast<float>(h + 0) * bin_size_h + roi_start_h));
                            int hend = static_cast<int>(ceil(static_cast<float>(h + 1) * bin_size_h + roi_start_h));

                            hstart = std::min<int>(std::max<int>(hstart, 0), height);
                            hend = std::min<int>(std::max<int>(hend, 0), height);
                            int wstart = static_cast<int>(floor(static_cast<float>(w + 0) * bin_size_w + roi_start_w));
                            int wend = static_cast<int>(ceil(static_cast<float>(w + 1) * bin_size_w + roi_start_w));

                            wstart = std::min<int>(std::max<int>(wstart, 0), width);
                            wend = std::min<int>(std::max<int>(wend, 0), width);

                            float bin_area = static_cast<float>((hend - hstart) * (wend - wstart));
                            if (bin_area) {
                                int gc = (c * group_size_ + h) * group_size_ + w;
                                const float *bottom_data =
                                        bottom_data_beginning + ((roi_batch_ind * channels + gc) * height * width);

                                float out_sum = 0.0f;
                                for (int hh = hstart; hh < hend; ++hh)
                                    for (int ww = wstart; ww < wend; ++ww)
                                        out_sum += bottom_data[hh * width + ww];

                                dst_data[index] = out_sum / bin_area;
                            }
                        } else if (mode_ == "bilinear") {
                            for (size_t bin_y = 0; bin_y < spatial_bins_y_; bin_y++) {
                                for (size_t bin_x = 0; bin_x < spatial_bins_x_; bin_x++) {
                                    float box_xmin = roi_start_w + (bin_x + 0) * (roi_width / spatial_bins_x_);
                                    float box_xmax = roi_start_w + (bin_x + 1) * (roi_width / spatial_bins_x_);
                                    float box_ymin = roi_start_h + (bin_y + 0) * (roi_height / spatial_bins_y_);
                                    float box_ymax = roi_start_h + (bin_y + 1) * (roi_height / spatial_bins_y_);

                                    size_t gc = c + (bin_y*spatial_bins_x_ + bin_x)*nc;
                                    size_t src_idx = (roi_batch_ind * channels + gc) * height * width;
                                    const float *bottom_data = bottom_data_beginning + src_idx;

                                    float height_scale = nh > 1 ? (box_ymax - box_ymin) * (height - 1) / (pooled_height_ - 1)
                                                                : 0.0f;
                                    float width_scale = nw > 1 ? (box_xmax - box_xmin) * (width - 1) / (pooled_width_ - 1)
                                                               : 0.0f;

                                    float in_y = nh > 1 ? (h * height_scale + box_ymin * (height - 1))
                                                        : 0.5f * (box_ymin + box_ymax) * (height - 1);
                                    float in_x = nw > 1 ? (w * width_scale + box_xmin * (width - 1))
                                                        : 0.5f * (box_xmin + box_xmax) * (width - 1);

                                    if (!(in_y < 0 || in_y > height - 1 || in_x < 0 || in_x > width - 1)) {
                                        int top_y_index = static_cast<int>(floorf(in_y));
                                        int bottom_y_index = static_cast<int>(ceilf(in_y));
                                        int left_x_index = static_cast<int>(floorf(in_x));
                                        int right_x_index = static_cast<int>(ceilf(in_x));

                                        if (right_x_index > width - 1)
                                            right_x_index = width - 1;

                                        if (bottom_y_index > height - 1)
                                            bottom_y_index = height - 1;

                                        const float top_left = bottom_data[top_y_index * width + left_x_index];
                                        const float top_right = bottom_data[top_y_index * width + right_x_index];
                                        const float bottom_left = bottom_data[bottom_y_index * width + left_x_index];
                                        const float bottom_right = bottom_data[bottom_y_index * width + right_x_index];

                                        const float top = top_left + (top_right - top_left) * (in_x - left_x_index);
                                        const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                                        dst_data[index] += top + (bottom - top) * (in_y - top_y_index);
                                    }
                                }
                            }
                            dst_data[index] /= num_bins;
                        }
                    }
                }
            }
        });

        for (int n = real_rois; n < nn; n++) {
            parallel_for3d(nc, nh, nw, [&](int c, int h, int w) {
                int index = n * nc * nh * nw + c * nh * nw + h * nw + w;
                dst_data[index] = 0.0f;
            });
        }

        return OK;
    }

private:
    size_t output_dim_ = 0;
    size_t group_size_ = 0;
    float spatial_scale_ = 0;
    size_t pooled_height_ = 0;
    size_t pooled_width_ = 0;
    size_t spatial_bins_x_ = 0;
    size_t spatial_bins_y_ = 0;
    std::string mode_ = "";

    int channels = 0;
    int height = 0;
    int width = 0;

    int nn = 0;
    int nc = 0;
    int nh = 0;
    int nw = 0;
};

REG_FACTORY_FOR(ImplFactory<PSROIPoolingImpl>, PSROIPooling);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
