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

class ExperimentalDetectronTopKROIsImpl: public ExtLayerBase {
private:
    // Inputs:
    //      rois, shape [n, 4]
    //      rois_probs, shape [n]
    // Outputs:
    //      top_rois, shape [max_rois, 4]

    const int INPUT_ROIS {0};
    const int INPUT_PROBS {1};

    const int OUTPUT_ROIS {0};

public:
    explicit ExperimentalDetectronTopKROIsImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            if (layer->insData[INPUT_ROIS].lock()->getTensorDesc().getDims().size() != 2 ||
                layer->insData[INPUT_PROBS].lock()->getTensorDesc().getDims().size() != 1)
                THROW_IE_EXCEPTION << "Unsupported shape of input blobs!";

            max_rois_num_ = layer->GetParamAsInt("max_rois", 0);

            addConfig(layer,
                      {DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)},
                      {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const int input_rois_num = inputs[INPUT_ROIS]->getTensorDesc().getDims()[0];
        const int top_rois_num = std::min(max_rois_num_, input_rois_num);

        auto *input_rois = inputs[INPUT_ROIS]->buffer().as<const float *>();
        auto *input_probs = inputs[INPUT_PROBS]->buffer().as<const float *>();
        auto *output_rois = outputs[OUTPUT_ROIS]->buffer().as<float *>();

        std::vector<size_t> idx(input_rois_num);
        iota(idx.begin(), idx.end(), 0);
        // FIXME. partial_sort is enough here.
        sort(idx.begin(), idx.end(), [&input_probs](size_t i1, size_t i2) {return input_probs[i1] > input_probs[i2];});

        for (int i = 0; i < top_rois_num; ++i) {
            std::memcpy(output_rois + 4 * i, input_rois + 4 * idx[i], 4 * sizeof(float));
        }

        return OK;
    }

private:
    int max_rois_num_;
};

REG_FACTORY_FOR(ImplFactory<ExperimentalDetectronTopKROIsImpl>, ExperimentalDetectronTopKROIs);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
