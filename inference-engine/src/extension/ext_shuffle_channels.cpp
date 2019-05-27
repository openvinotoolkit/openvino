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

class ShuffleChannelsImpl: public ExtLayerBase {
#define CNTR_SIZE 3

__inline size_t initter(size_t start, size_t size, size_t* counters, size_t* own_dims, size_t* ownStrides) {
    size_t i = start;
    size_t idx = 0;
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = i % own_dims[j];
        idx += counters[j] * ownStrides[j];
        i /= own_dims[j];
    }
    return idx;
}

__inline size_t updater(size_t idx, size_t size, size_t* counters, size_t* own_dims, size_t* ownStrides) {
    size_t i = 1;
    for (int j = size - 1; j >= 0; j--) {
        counters[j]++;
        if (counters[j] < own_dims[j]) {
            idx += ownStrides[j];
            break;
        } else {
            counters[j] = 0;
            i = 0;
        }
    }
    if (!i) {
        for (idx = 0; i < CNTR_SIZE; ++i)
            idx += counters[i] * ownStrides[i];
    }
    return idx;
}

public:
    explicit ShuffleChannelsImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            SizeVector src_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (src_dims.size() != dst_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            if (layer->insData[0].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only F32 is supported!";

            if (layer->outData[0]->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect output precision. Only F32 is supported!";

            int axis = layer->GetParamAsInt("axis", 1);
            if (axis < 0)
                axis += dst_dims.size();

            if (axis < 0 || axis >= static_cast<int>(dst_dims.size()))
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimensions and axis number!";

            size_t group = layer->GetParamAsUInt("group", 1);
            if (group == 0 || dst_dims[axis] % group)
                THROW_IE_EXCEPTION << layer->name << " Group parameter must evenly divide the channel dimension!";

            //  Find number of dictionaries, index range and data length
            own_dims[0] = 1;
            for (int i = 0; i < axis; i++)
                own_dims[0] *= dst_dims[i];

            for (size_t i = axis + 1; i < dst_dims.size(); i++)
                dataLength *= dst_dims[i];

            if (dataLength == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input parameters dimension!";

            own_dims[1] = dst_dims[axis] / group;
            own_dims[2] = group;
            ownStrides[0] = dst_dims[axis];
            ownStrides[1] = 1;
            ownStrides[2] = own_dims[1];
            work_amount_dst = ownStrides[0] * own_dims[0];

            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        const float *src_data = inputs[0]->cbuffer().as<const float *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        if (dataLength > 1) {
            //  Vectorized & Parallel
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0, src_idx = 0;
                size_t counters[CNTR_SIZE] = { 0 };
                splitter(work_amount_dst, nthr, ithr, start, end);
                src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
                for (size_t iwork = start, dst_idx = start * dataLength; iwork < end; ++iwork, dst_idx += dataLength) {
                    memcpy(&dst_data[dst_idx], &src_data[dataLength * src_idx], sizeof(float) * dataLength);
                    src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
                }
            });
        } else {
            //  Parallel
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0, src_idx = 0;
                size_t counters[CNTR_SIZE] = { 0 };
                splitter(work_amount_dst, nthr, ithr, start, end);
                src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    dst_data[iwork] = src_data[src_idx];
                    src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
                }
            });
        }

        return OK;
    }

private:
    size_t dataLength = 1;
    size_t work_amount_dst;
    size_t own_dims[CNTR_SIZE];
    size_t ownStrides[CNTR_SIZE];
};

REG_FACTORY_FOR(ImplFactory<ShuffleChannelsImpl>, ShuffleChannels);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
