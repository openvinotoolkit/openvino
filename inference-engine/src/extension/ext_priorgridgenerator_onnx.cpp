// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <algorithm>
#include <cassert>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

const int INPUT_PRIORS {0};
const int INPUT_FEATUREMAP {1};
const int INPUT_IMAGE {2};

const int OUTPUT_ROIS {0};

class ExperimentalDetectronPriorGridGeneratorImpl: public ExtLayerBase {
private:
    // Inputs:
    //      priors, shape [n, 4]
    //      [feature_map], shape [b, c, h, w]
    //      [im_data], shape [b, 3, im_h, im_w]
    // Outputs:
    //      priors_grid, shape [m, 4]

public:
    explicit ExperimentalDetectronPriorGridGeneratorImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() > 3 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[INPUT_PRIORS].lock()->getTensorDesc().getDims().size() != 2 ||
                    (layer->insData.size() > INPUT_FEATUREMAP &&
                     layer->insData[INPUT_FEATUREMAP].lock()->getTensorDesc().getDims().size() != 4) ||
                    (layer->insData.size() > INPUT_IMAGE &&
                     layer->insData[INPUT_IMAGE].lock()->getTensorDesc().getDims().size() != 4))
                THROW_IE_EXCEPTION << "Unsupported shape of input blobs!";

            grid_w_ = layer->GetParamAsInt("w", 0);
            grid_h_ = layer->GetParamAsInt("h", 0);
            stride_h_ = layer->GetParamAsFloat("stride_y", 0);
            stride_w_ = layer->GetParamAsFloat("stride_x", 0);

            addConfig(layer,
                      {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                      {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int num_priors_ = inputs[INPUT_PRIORS]->getTensorDesc().getDims()[0];
        assert(inputs[INPUT_PRIORS]->getTensorDesc().getDims()[1] == 4);

        // Execute
        const int layer_width = grid_w_ ? grid_w_ : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[3];
        const int layer_height = grid_h_ ? grid_h_ : inputs[INPUT_FEATUREMAP]->getTensorDesc().getDims()[2];
        const float step_w = stride_w_ ? stride_w_ : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[3]) / layer_width;
        const float step_h = stride_h_ ? stride_h_ : static_cast<float>(inputs[INPUT_IMAGE]->getTensorDesc().getDims()[2]) / layer_height;

        const auto *bottom_data_0 = inputs[0]->buffer().as<const float *>();
        auto *top_data_0 = outputs[OUTPUT_ROIS]->buffer().as<float *>();

        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int s = 0; s < num_priors_; ++s) {
                    top_data_0[0] = bottom_data_0[4 * s + 0] + step_w * (w + 0.5f);
                    top_data_0[1] = bottom_data_0[4 * s + 1] + step_h * (h + 0.5f);
                    top_data_0[2] = bottom_data_0[4 * s + 2] + step_w * (w + 0.5f);
                    top_data_0[3] = bottom_data_0[4 * s + 3] + step_h * (h + 0.5f);
                    top_data_0 += 4;
                }
            }
        }

        return OK;
    }

private:
    int grid_w_;
    int grid_h_;
    float stride_w_;
    float stride_h_;
};


REG_FACTORY_FOR(ImplFactory<ExperimentalDetectronPriorGridGeneratorImpl>, ExperimentalDetectronPriorGridGenerator);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
