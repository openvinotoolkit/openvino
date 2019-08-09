// Copyright (C) 2018-2019 Intel Corporation
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

class FillImpl: public ExtLayerBase {
public:
    explicit FillImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            SizeVector fill_dims = layer->insData[FILL_DIMS].lock()->getTensorDesc().getDims();
            if (fill_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Fill dimensions vector should be 1 dimension";

            if (layer->insData[FILL_DIMS].lock()->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Fill dimensions vector should be I32!";

            SizeVector value_dims = layer->insData[FILL_VALUE].lock()->getTensorDesc().getDims();
            if (value_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Value scalar should have 1 dimension";

            if (!(layer->insData[FILL_VALUE].lock()->getTensorDesc().getPrecision() == Precision::I32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::I32) &&
                !(layer->insData[FILL_VALUE].lock()->getTensorDesc().getPrecision() == Precision::FP32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::FP32)) {
                THROW_IE_EXCEPTION << layer->name <<
                    " 'Value' input scalars and output tensor should have same precision and only FP32 and I32 are supported!";
            }

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        int32_t* fill_dims = inputs[FILL_DIMS]->cbuffer().as<int32_t *>() +
                             inputs[FILL_DIMS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t fill_size = inputs[FILL_DIMS]->getTensorDesc().getDims()[0];
        SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();

        if (dst_dims.size() != fill_size) {
            if (resp) {
                std::string errorMsg = "Output tensor dimension mismatch";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return PARAMETER_MISMATCH;
        }

        size_t work_amount_dst = 1;
        for (size_t i = 0; i < dst_dims.size(); i++) {
            work_amount_dst *= fill_dims[i];
            if (static_cast<int>(dst_dims[i]) != fill_dims[i]) {
                if (resp) {
                    std::string errorMsg = "Output tensor dimension size mismatch";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }

        switch (outputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            float* dst_data = outputs[0]->cbuffer().as<float *>() +
                              outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            float value = (inputs[FILL_VALUE]->cbuffer().as<float *>() +
                           inputs[FILL_VALUE]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(work_amount_dst, nthr, ithr, start, end);
                std::fill_n(dst_data + start, end - start, value);
            });
        }
        break;
        case Precision::I32: {
            int32_t* dst_data = outputs[0]->cbuffer().as<int32_t *>() +
                                outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            int32_t value = (inputs[FILL_VALUE]->cbuffer().as<int32_t *>() +
                             inputs[FILL_VALUE]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(work_amount_dst, nthr, ithr, start, end);
                std::fill_n(dst_data + start, end - start, value);
            });
            return OK;
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect output precision. Only FP32 and I32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }

        return OK;
    }

private:
    const size_t FILL_DIMS = 0;
    const size_t FILL_VALUE = 1;
};

REG_FACTORY_FOR(ImplFactory<FillImpl>, Fill);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
