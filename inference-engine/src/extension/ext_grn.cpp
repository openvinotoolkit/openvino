// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class GRNImpl: public ExtLayerBase {
public:
    explicit GRNImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            bias = layer->GetParamAsFloat("bias");

            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        parallel_for3d(N, H, W, [&](int b, int h, int w) {
            double variance = 0;
            for (int c = 0; c < C; c++) {
                variance += std::pow(src_data[b*C*H*W + c*H*W + h*W + w], 2);
            }
            variance = std::pow(variance + bias, 0.5f);
            for (int c = 0; c < C; c++) {
                dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] / static_cast<float>(variance);
            }
        });
        return OK;
    }

private:
    float bias = 1.0f;
};

REG_FACTORY_FOR(ImplFactory<GRNImpl>, GRN);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
