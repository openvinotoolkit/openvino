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

class SqueezeImpl: public ExtLayerBase {
public:
    explicit SqueezeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            idx_dims = layer->insData[SQUEEZE_INDEXES].lock()->getTensorDesc().getDims();
            if (idx_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Index vector should be 1 dimension";

            if (layer->insData[SQUEEZE_INDEXES].lock()->getTensorDesc().getPrecision() != Precision::I32 &&
                layer->insData[SQUEEZE_INDEXES].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'indices_to_squeeze' input precision. Only FP32 and I32 are supported!";

            data_dims = layer->insData[SQUEEZE_DATA].lock()->getTensorDesc().getDims();
            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (data_dims.size() < dst_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            if (data_dims.size() <= idx_dims[0] && !(data_dims.size() == 1 && idx_dims[0] == 1))
                THROW_IE_EXCEPTION << layer->name << " Incompatible number of data dimensions and indexes vector length!";

            addConfig(layer, { { ConfLayout::PLN, false, 0 }, { ConfLayout::ANY, true } }, { { ConfLayout::PLN, false, 0 } });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        switch (inputs[SQUEEZE_INDEXES]->precision()) {
        case Precision::FP32: {
            float *idx_data = inputs[SQUEEZE_INDEXES]->cbuffer().as<float *>() +
                              inputs[SQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            for (size_t i = 0; i < idx_dims[0]; i++) {
                float axis = idx_data[i];
                if (axis < 0)
                    axis += data_dims.size();

                if (axis > static_cast<int>(data_dims.size())) {
                    if (resp) {
                        std::string errorMsg = "Index to squeeze exceeds data tensor dimension";
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return PARAMETER_MISMATCH;
                } else if (data_dims[static_cast<int>(axis)] != 1) {
                    if (resp) {
                        std::string errorMsg = "Index to squeeze of data tensor dimension is not 1";
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return PARAMETER_MISMATCH;
                }
            }
        }
        break;
        case Precision::I32: {
            int32_t *idx_data = inputs[SQUEEZE_INDEXES]->cbuffer().as<int32_t *>() +
                                inputs[SQUEEZE_INDEXES]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            for (size_t i = 0; i < idx_dims[0]; i++) {
                int32_t axis = idx_data[i];
                if (axis < 0)
                    axis += data_dims.size();

                if (axis > static_cast<int>(data_dims.size())) {
                    if (resp) {
                        std::string errorMsg = "Index to squeeze exceeds data tensor dimension";
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return PARAMETER_MISMATCH;
                } else if (data_dims[axis] != 1) {
                    if (resp) {
                        std::string errorMsg = "Index to squeeze of data tensor dimension is not 1";
                        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                    }
                    return PARAMETER_MISMATCH;
                }
            }
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect 'indices_to_squeeze' input precision. Only FP32 and I32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    const size_t SQUEEZE_DATA = 0;
    const size_t SQUEEZE_INDEXES = 1;

    SizeVector data_dims;
    SizeVector idx_dims;
};

REG_FACTORY_FOR(ImplFactory<SqueezeImpl>, Squeeze);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
