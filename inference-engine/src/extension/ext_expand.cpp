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

class ExpandImpl: public ExtLayerBase {
public:
    explicit ExpandImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            if (layer->insData.size() != 2)
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input edges!";

            SizeVector shape_dims = layer->insData[EXPAND_SHAPE].lock()->getTensorDesc().getDims();
            if (shape_dims.size() > 1)
                THROW_IE_EXCEPTION << layer->name << " Shape vector should be 1 dimension";

            if (layer->insData[EXPAND_SHAPE].lock()->getTensorDesc().getPrecision() != Precision::I32)
                THROW_IE_EXCEPTION << layer->name << " Shape vector should be I32!";

            if (!(layer->insData[EXPAND_INPUT].lock()->getTensorDesc().getPrecision() == Precision::I32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::I32) &&
                !(layer->insData[EXPAND_INPUT].lock()->getTensorDesc().getPrecision() == Precision::FP32 &&
                  layer->outData[0]->getTensorDesc().getPrecision() == Precision::FP32)) {
                THROW_IE_EXCEPTION << layer->name <<
                    " Input and output tensors should have same precision and only FP32 and I32 are supported!";
            }

            src_dims = layer->insData[EXPAND_INPUT].lock()->getTensorDesc().getDims();
            srcStrides = layer->insData[EXPAND_INPUT].lock()->getTensorDesc().getBlockingDesc().getStrides();
            addConfig(layer, { DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN) },
                             { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        int32_t* shape_dims = inputs[EXPAND_SHAPE]->cbuffer().as<int32_t *>() +
                              inputs[EXPAND_SHAPE]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t shape_size = (inputs[EXPAND_SHAPE]->getTensorDesc().getDims())[0];
        SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();

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

        size_t i;
        for (i = 0; i < dst_dims.size(); i++) {
            if (static_cast<int>(dst_dims[i]) != shape_dims[i]) {
                if (resp) {
                    std::string errorMsg = "Output tensor dimension size mismatch";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }

        size_t prefix_size = dst_dims.size() - src_dims.size();
        for (i = 0; i < src_dims.size(); i++) {
            if (src_dims[i] != 1 &&
                    static_cast<int>(src_dims[i]) != shape_dims[i + prefix_size]) {
                if (resp) {
                    std::string errorMsg = "In/Output corresponding dimension must have the same value, or Input dimension is equal to 1";
                    errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
                }
                return PARAMETER_MISMATCH;
            }
        }

        InferenceEngine::SizeVector dstStrides = outputs[0]->getTensorDesc().getBlockingDesc().getStrides();
        InferenceEngine::SizeVector src_aligned(dst_dims.size());
        InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
        for (i = 0; i < dst_dims.size(); i++) {
            if (i < prefix_size) {
                src_aligned[i] = 1;
                srcStrides_aligned[i] = srcStrides[0];
            } else {
                src_aligned[i] = src_dims[i - prefix_size];
                srcStrides_aligned[i] = srcStrides[i - prefix_size];
            }
        }

        size_t work_amount_dst = dstStrides[0] * dst_dims[0];

        switch (outputs[0]->precision()) {
        case Precision::FP32: {
            const float *src_data = inputs[EXPAND_INPUT]->cbuffer().as<const float *>() +
                                    inputs[EXPAND_INPUT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            float* dst_data = outputs[0]->cbuffer().as<float *>() +
                              outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t i, src_idx, start = 0, end = 0;
                SizeVector counters(dst_dims.size(), 0);
                splitter(work_amount_dst, nthr, ithr, start, end);
                for (int j = dst_dims.size() - 1, i = start; j >= 0; j--) {
                    counters[j] = i % dst_dims[j];
                    i /= dst_dims[j];
                }
                for (size_t iwork = start; iwork < end; ++iwork) {
                    for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
                        src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

                    dst_data[iwork] = src_data[src_idx];

                    for (int j = dst_dims.size() - 1; j >= 0; j--) {
                        counters[j] = (counters[j] + 1) % dst_dims[j];
                        if (counters[j] != 0) break;
                    }
                }
            });
        }
        break;
        case Precision::I32: {
            const int32_t *src_data = inputs[EXPAND_INPUT]->cbuffer().as<const int32_t *>() +
                                      inputs[EXPAND_INPUT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
            int32_t* dst_data = outputs[0]->cbuffer().as<int32_t *>() +
                                outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t i, src_idx, start = 0, end = 0;
                SizeVector counters(dst_dims.size(), 0);
                splitter(work_amount_dst, nthr, ithr, start, end);
                for (int j = dst_dims.size() - 1, i = start; j >= 0; j--) {
                    counters[j] = i % dst_dims[j];
                    i /= dst_dims[j];
                }
                for (size_t iwork = start; iwork < end; ++iwork) {
                    for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
                        src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

                    dst_data[iwork] = src_data[src_idx];

                    for (int j = dst_dims.size() - 1; j >= 0; j--) {
                        counters[j] = (counters[j] + 1) % dst_dims[j];
                        if (counters[j] != 0) break;
                    }
                }
            });
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
    const size_t EXPAND_INPUT = 0;
    const size_t EXPAND_SHAPE = 1;

    SizeVector src_dims;
    SizeVector srcStrides;
};

REG_FACTORY_FOR(ImplFactory<ExpandImpl>, Expand);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
