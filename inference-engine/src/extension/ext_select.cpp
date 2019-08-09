// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class SelectImpl: public ExtLayerBase {
    enum {condition, then_, else_, numOfInputs};

public:
    explicit SelectImpl(const CNNLayer* layer) {
        try {
            if (numOfInputs != layer->insData.size() || 1 != layer->outData.size()) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";
            }

            auto conditionPrecision = layer->insData[condition].lock()->getTensorDesc().getPrecision();

            if (Precision::I32 != conditionPrecision && Precision::FP32 != conditionPrecision) {
                THROW_IE_EXCEPTION << layer->name << " Incorrect condition tensor precision: " << conditionPrecision << ". Should be I32 or FP32";
            }

            addConfig(layer, {{ConfLayout::PLN, false},
                              {ConfLayout::PLN, false},
                              {ConfLayout::PLN, false}},
                             {{ConfLayout::PLN, false}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *) noexcept override {
        const int32_t *conditionData = inputs[condition]->cbuffer().as<const int32_t *>();

        const float *thenData = inputs[then_]->cbuffer().as<const float *>();

        const float *elseData = inputs[else_]->cbuffer().as<const float *>();

        float* dstData = outputs[0]->cbuffer().as<float *>();
        enum {N, C, H, W, Dims};
        int dim[Dims] = {1, 1, 1, 1};
        int cdim[Dims] = {1, 1, 1, 1};

        SizeVector dims = inputs[then_]->getTensorDesc().getDims();
        std::copy(std::begin(dims), std::end(dims), std::begin(dim) + (Dims - dims.size()));

        SizeVector cDims = inputs[condition]->getTensorDesc().getDims();
        std::copy(std::begin(cDims), std::end(cDims), std::begin(cdim) + (Dims - cDims.size()));

        parallel_for3d(dim[N], dim[H], dim[W], [&](int b, int h, int w) {
            for (int c = 0; c < dim[C]; c++) {
                        dstData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
                = conditionData[(b % cdim[N])*cdim[C]*cdim[H]*cdim[W] + (c % cdim[C])*cdim[H]*cdim[W] + (h % cdim[H])*cdim[W] + (w % cdim[W])]
                ?      thenData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w]
                :      elseData[b*dim[C]*dim[H]*dim[W] + c*dim[H]*dim[W] + h*dim[W] + w];
            }
        });
        return OK;
    }
};


REG_FACTORY_FOR(ImplFactory<SelectImpl>, Select);
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
