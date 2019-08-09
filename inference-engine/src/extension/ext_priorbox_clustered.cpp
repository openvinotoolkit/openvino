// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <algorithm>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PriorBoxClusteredImpl: public ExtLayerBase {
public:
    explicit PriorBoxClusteredImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[0].lock()->getTensorDesc().getDims().size() != 4 ||
                    layer->insData[1].lock()->getTensorDesc().getDims().size() != 4)
                THROW_IE_EXCEPTION << "PriorBoxClustered supports only 4D blobs!";

            widths_ = layer->GetParamAsFloats("width", {});
            heights_ = layer->GetParamAsFloats("height", {});
            clip_ = layer->GetParamAsInt("clip");
            variance_ = layer->GetParamAsFloats("variance", {});
            img_h_ = layer->GetParamAsInt("img_h", 0);
            img_w_ = layer->GetParamAsInt("img_w", 0);
            step_ = layer->GetParamAsFloat("step", 0);
            step_h_ = layer->GetParamAsFloat("step_h", 0);
            step_w_ = layer->GetParamAsFloat("step_w", 0);
            offset_ = layer->GetParamAsFloat("offset");

            addConfig(layer, {{ConfLayout::PLN, true}, {ConfLayout::PLN, true}}, {{ConfLayout::PLN, true}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode init(LayerConfig& config, ResponseDesc *resp) noexcept override {
        return OK;
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        int num_priors_ = widths_.size();

        if (variance_.empty())
            variance_.push_back(0.1f);

        // Execute
        const int layer_width = inputs[0]->getTensorDesc().getDims()[3];
        const int layer_height = inputs[0]->getTensorDesc().getDims()[2];

        int img_width = img_w_ == 0 ? inputs[1]->getTensorDesc().getDims()[3] : img_w_;
        int img_height = img_h_ == 0 ? inputs[1]->getTensorDesc().getDims()[2] : img_h_;

        float step_w = step_w_ == 0 ? step_ : step_w_;
        float step_h = step_h_ == 0 ? step_ : step_h_;
        if (step_w == 0 && step_h == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        }

        auto *top_data_0 = outputs[0]->buffer().as<float *>();
        float *top_data_1 = top_data_0 + outputs[0]->getTensorDesc().getDims()[2];
        int var_size = variance_.size();

        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                float center_x = (w + offset_) * step_w;
                float center_y = (h + offset_) * step_h;

                for (int s = 0; s < num_priors_; ++s) {
                    float box_width = widths_[s];
                    float box_height = heights_[s];

                    float xmin = (center_x - box_width / 2.0f) / img_width;
                    float ymin = (center_y - box_height / 2.0f) / img_height;
                    float xmax = (center_x + box_width / 2.0f) / img_width;
                    float ymax = (center_y + box_height / 2.0f) / img_height;

                    if (clip_) {
                        xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                        ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                        xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                        ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                    }

                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 0] = xmin;
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 1] = ymin;
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 2] = xmax;
                    top_data_0[h * layer_width * num_priors_ * 4 + w * num_priors_ * 4 + s * 4 + 3] = ymax;

                    for (int j = 0; j < var_size; j++)
                        top_data_1[h * layer_width * num_priors_ * var_size + w * num_priors_ * var_size +
                                   s * var_size +
                                   j] = variance_[j];
                }
            }
        }
        return OK;
    }

private:
    std::vector<float> widths_;
    std::vector<float> heights_;
    std::vector<float> variance_;
    int clip_;
    int img_h_;
    int img_w_;
    float step_;
    float step_h_;
    float step_w_;
    float offset_;
};

REG_FACTORY_FOR(ImplFactory<PriorBoxClusteredImpl>, PriorBoxClustered);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
