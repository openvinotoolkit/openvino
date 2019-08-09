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

class RangeImpl: public ExtLayerBase {
public:
    explicit RangeImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 3)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            SizeVector start_dims = layer->insData[RANGE_START].lock()->getTensorDesc().getDims();
            if (start_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Start scalar should have 1 dimension";

            SizeVector limit_dims = layer->insData[RANGE_LIMIT].lock()->getTensorDesc().getDims();
            if (limit_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Limit scalar should have 1 dimension";

            SizeVector delta_dims = layer->insData[RANGE_DELTA].lock()->getTensorDesc().getDims();
            if (delta_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Delta scalar should have 1 dimension";

            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (dst_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Output vector should have 1 dimension";

            if (!(layer->insData[RANGE_START].lock()->getTensorDesc().getPrecision() == Precision::I32 &&
                  layer->insData[RANGE_LIMIT].lock()->getTensorDesc().getPrecision() == Precision::I32 &&
                  layer->insData[RANGE_DELTA].lock()->getTensorDesc().getPrecision() == Precision::I32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::I32) &&
                !(layer->insData[RANGE_START].lock()->getTensorDesc().getPrecision() == Precision::FP32 &&
                  layer->insData[RANGE_LIMIT].lock()->getTensorDesc().getPrecision() == Precision::FP32 &&
                  layer->insData[RANGE_DELTA].lock()->getTensorDesc().getPrecision() == Precision::FP32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::FP32)) {
                THROW_IE_EXCEPTION << layer->name <<
                    " 'Start', 'Limit', 'Delta' input scalars and output tensor should have same precision" <<
                    "and only FP32 and I32 are supported!";
            }

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        StatusCode retcode = OK;
        switch (outputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32: {
            retcode = range((inputs[RANGE_START]->cbuffer().as<float *>() +
                             inputs[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_LIMIT]->cbuffer().as<float *>() +
                             inputs[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_DELTA]->cbuffer().as<float *>() +
                             inputs[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outputs[0]);
        }
        break;
        case Precision::I32: {
            retcode = range((inputs[RANGE_START]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_LIMIT]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                            (inputs[RANGE_DELTA]->cbuffer().as<int32_t *>() +
                             inputs[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outputs[0]);
        }
        break;
        default:
            if (resp) {
                std::string errorMsg = "Incorrect output precision. Only FP32 and I32 are supported!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            retcode = GENERAL_ERROR;
        }
        if (resp && retcode == PARAMETER_MISMATCH) {
            std::string errorMsg = "Range indexes exceeds data tensor dimension";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        }
        return retcode;
    }

private:
    const size_t RANGE_START = 0;
    const size_t RANGE_LIMIT = 1;
    const size_t RANGE_DELTA = 2;

    template <typename data_t>
    StatusCode range(data_t start, data_t limit, data_t delta, Blob::Ptr output);
};

template <typename data_t>
StatusCode RangeImpl::range(data_t start, data_t limit, data_t delta, Blob::Ptr output) {
    size_t dst_size = (output->getTensorDesc().getDims())[0];
    data_t* dst_data = output->cbuffer().as<data_t *>() +
                       output->getTensorDesc().getBlockingDesc().getOffsetPadding();
    size_t work_amount_dst = static_cast<size_t>(std::floor(std::abs((limit - start) / delta)));
    if (work_amount_dst != dst_size)
        return PARAMETER_MISMATCH;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t iwork = 0, end = 0;
        splitter(work_amount_dst, nthr, ithr, iwork, end);
        data_t dst_value = start + iwork * delta;

        for (; iwork < end; ++iwork, dst_value += delta) {
            dst_data[iwork] = dst_value;
        }
    });
    return OK;
}
REG_FACTORY_FOR(ImplFactory<RangeImpl>, Range);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
