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

class ReverseSequenceImpl: public ExtLayerBase {
public:
    explicit ReverseSequenceImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 2 || layer->outData.size() != 1)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            src_dims = layer->insData[REVERSESEQUENCE_DATA].lock()->getTensorDesc().getDims();
            SizeVector seq_lengths_dims = layer->insData[REVERSESEQUENCE_LENGTHS].lock()->getTensorDesc().getDims();
            if (layer->insData[REVERSESEQUENCE_LENGTHS].lock()->getTensorDesc().getPrecision() != Precision::I32 &&
                layer->insData[REVERSESEQUENCE_LENGTHS].lock()->getTensorDesc().getPrecision() != Precision::FP32)
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'seq_lengths' input precision. Only FP32 and I32 are supported!";
            if (seq_lengths_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Seq_lengths vector should be 1 dimension";

            SizeVector dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (src_dims.size() != dst_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output sizes!";

            for (size_t i = 0; i < dst_dims.size(); i++) {
                if (src_dims[i] != dst_dims[i])
                    THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimension!";
            }

            seq_axis = layer->GetParamAsInt("seq_axis", 1);
            if (seq_axis < 0)
                seq_axis += src_dims.size();

            if (seq_axis < 0 || seq_axis >= static_cast<int>(src_dims.size()))
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'seq_axis' parameters dimensions and axis number!";

            batch_axis = layer->GetParamAsInt("batch_axis", 0);
            if (batch_axis < 0)
                batch_axis += src_dims.size();

            if (batch_axis < 0 || batch_axis >= static_cast<int>(src_dims.size()))
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'batch_axis' parameters dimensions and axis number!";

            if (seq_lengths_dims[0] != dst_dims[batch_axis])
                THROW_IE_EXCEPTION << layer->name << " Incorrect 'seq_lengths_dims' parameters dimension!";

            srcStrides = layer->insData[REVERSESEQUENCE_DATA].lock()->getTensorDesc().getBlockingDesc().getStrides();
            work_amount_dst = srcStrides[0] * src_dims[0];

            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        size_t i;
        const float *src_data = inputs[REVERSESEQUENCE_DATA]->cbuffer().as<const float *>() +
                                inputs[REVERSESEQUENCE_DATA]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
                          outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        switch (inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                float *seq_lengths_data = inputs[REVERSESEQUENCE_LENGTHS]->cbuffer().as<float *>() +
                                          inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                for (i = 0; i < src_dims[batch_axis]; i++) {
                    if (static_cast<int32_t>(seq_lengths_data[i]) > static_cast<int>(src_dims[seq_axis])) {
                        if (resp) {
                            std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return PARAMETER_MISMATCH;
                    }
                }

                parallel_nt(0, [&](const int ithr, const int nthr) {
                    size_t i, start = 0, end = 0, src_idx = 0;
                    SizeVector counters(src_dims.size(), 0);
                    splitter(work_amount_dst, nthr, ithr, start, end);
                    for (int j = src_dims.size() - 1, i = start; j >= 0; j--) {
                        counters[j] = i % src_dims[j];
                        i /= src_dims[j];
                    }

                    for (size_t iwork = start; iwork < end; ++iwork) {
                        for (i = 0, src_idx = 0; i < src_dims.size(); ++i) {
                            size_t idx = counters[i];
                            if (static_cast<int>(i) == seq_axis &&
                                    static_cast<int>(idx) < static_cast<int32_t>(seq_lengths_data[counters[batch_axis]])) {
                                idx = static_cast<int32_t>(seq_lengths_data[counters[batch_axis]]) - idx - 1;
                            }
                            src_idx += idx * srcStrides[i];
                        }
                        dst_data[iwork] = src_data[src_idx];
                        for (int j = src_dims.size() - 1; j >= 0; j--) {
                            counters[j] = (counters[j] + 1) % src_dims[j];
                            if (counters[j] != 0) break;
                        }
                    }
                });
            }
            break;
            case Precision::I32: {
                int32_t *seq_lengths_data = inputs[REVERSESEQUENCE_LENGTHS]->cbuffer().as<int32_t *>() +
                                            inputs[REVERSESEQUENCE_LENGTHS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                for (i = 0; i < src_dims[batch_axis]; i++) {
                    if (seq_lengths_data[i] > static_cast<int>(src_dims[seq_axis])) {
                        if (resp) {
                            std::string errorMsg = "Incorrect input 'seq_lengths' values!";
                            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                        }
                        return PARAMETER_MISMATCH;
                    }
                }

                parallel_nt(0, [&](const int ithr, const int nthr) {
                    size_t i, start = 0, end = 0, src_idx = 0;
                    SizeVector counters(src_dims.size(), 0);
                    splitter(work_amount_dst, nthr, ithr, start, end);
                    for (int j = src_dims.size() - 1, i = start; j >= 0; j--) {
                        counters[j] = i % src_dims[j];
                        i /= src_dims[j];
                    }

                    for (size_t iwork = start; iwork < end; ++iwork) {
                        for (i = 0, src_idx = 0; i < src_dims.size(); ++i) {
                            size_t idx = counters[i];
                            if (static_cast<int>(i) == seq_axis &&
                                    static_cast<int>(idx) < seq_lengths_data[counters[batch_axis]]) {
                                idx = seq_lengths_data[counters[batch_axis]] - idx - 1;
                            }
                            src_idx += idx * srcStrides[i];
                        }
                        dst_data[iwork] = src_data[src_idx];
                        for (int j = src_dims.size() - 1; j >= 0; j--) {
                            counters[j] = (counters[j] + 1) % src_dims[j];
                            if (counters[j] != 0) break;
                        }
                    }
                });
            }
            break;
            default:
                return GENERAL_ERROR;
        }

        return OK;
    }

private:
    const size_t REVERSESEQUENCE_DATA = 0;
    const size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;
    SizeVector src_dims;
    SizeVector srcStrides;
    size_t work_amount_dst;
};

REG_FACTORY_FOR(ImplFactory<ReverseSequenceImpl>, ReverseSequence);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
