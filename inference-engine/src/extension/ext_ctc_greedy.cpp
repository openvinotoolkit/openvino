// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class CTCGreedyDecoderImpl: public ExtLayerBase {
public:
    explicit CTCGreedyDecoderImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            std::vector<DataConfigurator> inps;
            for (const auto &in : layer->insData)
                inps.emplace_back(ConfLayout::PLN);
            addConfig(layer, inps, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if ((inputs.size() != 1 && inputs.size() != 2) || outputs.empty()) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        const float* probabilities = inputs[0]->buffer();
        const float* sequence_indicators = inputs[1]->buffer();
        float* output_sequences = outputs[0]->buffer();

        size_t T_ = inputs[0]->getTensorDesc().getDims()[0];
        size_t N_ = inputs[0]->getTensorDesc().getDims()[1];
        size_t C_ = inputs[0]->getTensorDesc().getDims()[2];

        // Fill output_sequences with -1
        for (size_t ii = 0; ii < T_*N_; ii++) {
            output_sequences[ii] = -1;
        }

        for (int n = 0; n < N_; ++n) {
            int prev_class_idx = -1;
            size_t output_index = n*T_;

            for (int t = 0; /* check at end */; ++t) {
                // get maximum probability and its index
                int max_class_idx = 0;

                const float* probs = probabilities + t*C_*N_ + n*C_;
                float max_prob = probs[0];
                ++probs;

                for (int c = 1; c < C_; ++c, ++probs) {
                    if (*probs > max_prob) {
                        max_class_idx = c;
                        max_prob = *probs;
                    }
                }

                if (max_class_idx < C_-1 && max_class_idx != prev_class_idx) {
                    output_sequences[output_index] =  max_class_idx;
                    output_index++;
                }

                prev_class_idx = max_class_idx;

                if (t + 1 == T_ || sequence_indicators[(t + 1)*N_ + n] == 0) {
                    break;
                }
            }
        }
        return OK;
    }
};

REG_FACTORY_FOR(ImplFactory<CTCGreedyDecoderImpl>, CTCGreedyDecoder);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
