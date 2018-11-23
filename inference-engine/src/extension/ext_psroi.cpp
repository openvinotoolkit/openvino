// Copyright (C) 2018 Intel Corporation
//
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

        parallel_for(real_rois, [&](int n) {
            const float* bottom_rois = bottom_rois_beginning + n * 5;
            int roi_batch_ind = static_cast<int>(bottom_rois[0]);
            float roi_start_w = static_cast<float>(round(bottom_rois[1])) * spatial_scale_;
            float roi_start_h = static_cast<float>(round(bottom_rois[2])) * spatial_scale_;
            float roi_end_w   = static_cast<float>(round(bottom_rois[3]) + 1.0f) * spatial_scale_;
            float roi_end_h   = static_cast<float>(round(bottom_rois[4]) + 1.0f) * spatial_scale_;

            // Force too small ROIs to be 1x1
            float roi_width  = std::max<float>(roi_end_w - roi_start_w, 0.1f);  // avoid 0
            float roi_height = std::max<float>(roi_end_h - roi_start_h, 0.1f);

            float bin_size_h = roi_height / static_cast<float>(pooled_height_);
            float bin_size_w = roi_width  / static_cast<float>(pooled_width_);

            for (int c = 0; c < nc; c++) {
                for (int h = 0; h < nh; h++) {
                    int hstart = floor(static_cast<float>(h + 0) * bin_size_h + roi_start_h);
                    int hend = ceil(static_cast<float>(h + 1) * bin_size_h + roi_start_h);

                    hstart = std::min<int>(std::max<int>(hstart, 0), height);
                    hend = std::min<int>(std::max<int>(hend, 0), height);

                    for (int w = 0; w < nw; w++) {
                        int index = n * nc * nh * nw + c * nh * nw + h * nw + w;
                        dst_data[index] = 0.0f;

                        int wstart = floor(static_cast<float>(w + 0) * bin_size_w + roi_start_w);
                        int wend = ceil(static_cast<float>(w + 1) * bin_size_w + roi_start_w);

                        wstart = std::min<int>(std::max<int>(wstart, 0), width);
                        wend = std::min<int>(std::max<int>(wend, 0), width);

                        float bin_area = (hend - hstart) * (wend - wstart);
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
