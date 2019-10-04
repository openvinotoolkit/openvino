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
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class BroadcastImpl: public ExtLayerBase {
public:
    explicit BroadcastImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            SizeVector shape_dims = layer->insData[BROADCAST_SHAPE].lock()->getTensorDesc().getDims();
            if (shape_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Shape vector should be 1 dimension";

            data_size = layer->insData[BROADCAST_INPUT].lock()->getTensorDesc().getPrecision().size();
            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        size_t shape_size = (inputs[BROADCAST_SHAPE]->getTensorDesc().getDims())[0];
        SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();
        SizeVector src_dims = inputs[BROADCAST_INPUT]->getTensorDesc().getDims();
        SizeVector srcStrides = inputs[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getStrides();

        if (!src_dims.size())
            src_dims = SizeVector(1, 1);
        if (!srcStrides.size())
            srcStrides = SizeVector(1, 1);

        if (dst_dims.size() != shape_size) {
            if (resp) {
                std::string errorMsg = "Output tensor dimension mismatch";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return PARAMETER_MISMATCH;
        }

        if (src_dims.size() > dst_dims.size()) {
            if (resp) {
                std::string errorMsg = "Output tensor dimension is smaller then input tensor dimension";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return PARAMETER_MISMATCH;
        }

        InferenceEngine::SizeVector dstStrides = outputs[0]->getTensorDesc().getBlockingDesc().getStrides();
        InferenceEngine::SizeVector src_aligned(dst_dims.size());
        InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
        size_t prefix_size = dst_dims.size() - src_dims.size();
        for (size_t i = 0; i < dst_dims.size(); i++) {
            if (i < prefix_size) {
                src_aligned[i] = 1;
                srcStrides_aligned[i] = srcStrides[0];
            } else {
                src_aligned[i] = src_dims[i - prefix_size];
                srcStrides_aligned[i] = srcStrides[i - prefix_size];
            }
        }

        size_t work_amount_dst = dstStrides[0] * dst_dims[0];
        const uint8_t *src_data = inputs[BROADCAST_INPUT]->cbuffer().as<const uint8_t *>() +
                                inputs[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t* dst_data = outputs[0]->cbuffer().as<uint8_t *>() +
                          outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t i, src_idx, start = 0, end = 0;
            SizeVector counters(dst_dims.size(), 0);
            splitter(work_amount_dst, nthr, ithr, start, end);
            for (int j = dst_dims.size() - 1, i = start; j >= 0; j--) {
                counters[j] = i % dst_dims[j];
                i /= dst_dims[j];
            }
            for (size_t iwork = start * data_size; iwork < end * data_size; iwork += data_size) {
                for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
                    src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

                simple_copy(&dst_data[iwork], data_size, &src_data[src_idx * data_size], data_size);

                for (int j = dst_dims.size() - 1; j >= 0; j--) {
                    counters[j] = (counters[j] + 1) % dst_dims[j];
                    if (counters[j] != 0) break;
                }
            }
        });

        return OK;
    }

private:
    const size_t BROADCAST_INPUT = 0;
    const size_t BROADCAST_SHAPE = 1;

    size_t data_size = 1;
};

REG_FACTORY_FOR(ImplFactory<BroadcastImpl>, Broadcast);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
