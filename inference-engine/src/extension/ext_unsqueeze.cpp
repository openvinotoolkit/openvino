// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ext_list.hpp"
#include "ext_base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class UnsqueezeImpl: public ExtLayerBase {
public:
    explicit UnsqueezeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            idx_dims = layer->insData[UNSQUEEZE_INDEXES].lock()->getTensorDesc().getDims();
            data_dims = layer->insData[UNSQUEEZE_DATA].lock()->getTensorDesc().getDims();
            if (idx_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Index vector should be 1 dimension";

            if (layer->insData[UNSQUEEZE_INDEXES].lock()->getTensorDesc().getPrecision() != Precision::I32 &&
                layer->insData[UNSQUEEZE_INDEXES].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'indices_to_squeeze' input precision. Only FP32 and I32 are supported!";

            addConfig(layer, { { ConfLayout::PLN, false, 0 }, { ConfLayout::ANY, true } }, { { ConfLayout::PLN, false, 0 } });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        InferenceEngine::SizeVector data_dims = inputs[UNSQUEEZE_DATA]->getTensorDesc().getDims();
        InferenceEngine::SizeVector idx_dims = inputs[UNSQUEEZE_INDEXES]->getTensorDesc().getDims();

        switch (inputs[UNSQUEEZE_INDEXES]->precision()) {
        case Precision::FP32: {
            float *idx_data = inputs[UNSQUEEZE_INDEXES]->cbuffer().as<float *>() +
                              inputs[UNSQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();

            size_t max = data_dims.size();
            for (size_t i = 0; i < idx_dims[0]; i++) {
                size_t axis = static_cast<size_t>(idx_data[i]);
                if (axis > max) max = axis;
            }
            max++;

            if ((idx_dims[0] + data_dims.size()) < max) {
                if (resp) {
                    std::string errorMsg = "Indices_to_set for unsqueeze layer is out of tensor dimension";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }
        break;
        case Precision::I32: {
            int32_t *idx_data = inputs[UNSQUEEZE_INDEXES]->cbuffer().as<int32_t *>() +
                                inputs[UNSQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            size_t max = data_dims.size();
            for (size_t i = 0; i < idx_dims[0]; i++) {
                size_t axis = static_cast<size_t>(idx_data[i]);
                if (axis > max) max = axis;
            }
            max++;

            if ((idx_dims[0] + data_dims.size()) < max) {
                if (resp) {
                    std::string errorMsg = "Indices_to_set for unsqueeze layer is out of tensor dimension";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect 'indices_to_set' input precision. Only FP32 and I32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    const size_t UNSQUEEZE_DATA = 0;
    const size_t UNSQUEEZE_INDEXES = 1;

    SizeVector data_dims;
    SizeVector idx_dims;
};

REG_FACTORY_FOR(ImplFactory<UnsqueezeImpl>, Unsqueeze);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
