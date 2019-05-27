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

class SpaceToDepthImpl: public ExtLayerBase {
#define CNTR_SIZE 5

public:
    explicit SpaceToDepthImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            SizeVector src_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            if (src_dims.size() < 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input dimensions!";
            if (layer->insData[0].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect input precision. Only F32 is supported!";

            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (dst_dims.size() < 3)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of output dimensions!";
            if (layer->outData[0]->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect output precision. Only F32 is supported!";

            size_t block_size = layer->GetParamAsUInt("block_size", 1);
            if (block_size == 0)
                THROW_IE_EXCEPTION << layer->name << " Incorrect block_size parameter is zero!";

            if (dst_dims[dst_dims.size() - 3] % (block_size * block_size))
                THROW_IE_EXCEPTION << layer->name << " block_size parameter is incompatible with input tensor Color dimension size!";

            if (src_dims.size() > 2 && dst_dims[dst_dims.size() - 3] != (src_dims[src_dims.size() - 3] * block_size * block_size))
                THROW_IE_EXCEPTION << layer->name << " Input/Output tensor Color dimension is incompatible with block_size!";

            if (src_dims[src_dims.size() - 2] != (dst_dims[dst_dims.size() - 2] * block_size))
                THROW_IE_EXCEPTION << layer->name << " Input/Output tensor Height dimension is incompatible with block_size!";

            if (src_dims[src_dims.size() - 1] != (dst_dims[dst_dims.size() - 1] * block_size))
                THROW_IE_EXCEPTION << layer->name << " Input/Output tensor Width dimension is incompatible with block_size!";

            own_dims[0] = 1;
            for (size_t i = 0; i < (dst_dims.size() - 3); i++)
                own_dims[0] *= dst_dims[i];
            own_dims[1] = dst_dims[dst_dims.size() - 2];
            own_dims[2] = dst_dims[dst_dims.size() - 3] / block_size;
            own_dims[3] = dst_dims[dst_dims.size() - 1];
            own_dims[4] = block_size;

            size_t C = dst_dims[dst_dims.size() - 2] * dst_dims[dst_dims.size() - 1];
            ownStrides[0] = dst_dims[dst_dims.size() - 3] * C;
            ownStrides[1] = dst_dims[dst_dims.size() - 1];
            ownStrides[2] = block_size * C;
            ownStrides[3] = 1;
            ownStrides[4] = C;
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

        //  Parallel
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t i, start = 0, end = 0, dst_idx = 0;
            size_t counters[CNTR_SIZE] = { 0 };
            splitter(work_amount_dst, nthr, ithr, start, end);
            i = start;
            for (int j = CNTR_SIZE - 1; j >= 0; j--) {
                counters[j] = i % own_dims[j];
                dst_idx += counters[j] * ownStrides[j];
                i /= own_dims[j];
            }

            for (size_t iwork = start, i = 1; iwork < end; ++iwork) {
                dst_data[dst_idx] = src_data[iwork];
                for (int j = CNTR_SIZE - 1; j >= 0; j--) {
                    counters[j]++;
                    if (counters[j] < own_dims[j]) {
                        dst_idx += ownStrides[j];
                        break;
                    } else {
                        counters[j] = i = 0;
                    }
                }
                if (!i) {
                    for (dst_idx = 0; i < CNTR_SIZE; ++i)
                        dst_idx += counters[i] * ownStrides[i];
                }
            }
        });

        return OK;
    }

private:
    size_t work_amount_dst;
    size_t own_dims[CNTR_SIZE];
    size_t ownStrides[CNTR_SIZE];
};

REG_FACTORY_FOR(ImplFactory<SpaceToDepthImpl>, SpaceToDepth);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
