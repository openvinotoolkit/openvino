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

class PadImpl: public ExtLayerBase {
public:
    explicit PadImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.empty() || layer->outData.empty())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output edges!";

            pads_begin = layer->GetParamAsUInts("pads_begin");
            std::vector<unsigned int> pads_end = layer->GetParamAsUInts("pads_end");

            src_dims = layer->insData[0].lock()->getTensorDesc().getDims();
            dst_dims = layer->outData[0]->getTensorDesc().getDims();
            if (src_dims.size() != dst_dims.size() || pads_begin.size() != src_dims.size())
                THROW_IE_EXCEPTION << layer->name << " Incorrect number of input/output dimensions!";

            std::string pad_mode = layer->GetParamAsString("pad_mode");
            if (pad_mode == "constant") {
                padMode = CONSTANT;
            } else if (pad_mode == "edge") {
                padMode = EDGE;
            } else if (pad_mode == "reflect") {
                padMode = REFLECT;
                for (size_t i = 0; i < src_dims.size(); i++) {
                    if ((src_dims[i] - 1) < pads_begin[i] || (src_dims[i] - 1) < pads_end[i])
                        THROW_IE_EXCEPTION << layer->name << " Incorrect pads_begin or pads_end for 'reflect' pad mode";
                }
            } else if (pad_mode == "symmetric") {
                padMode = SYMMETRIC;
                for (size_t i = 0; i < src_dims.size(); i++) {
                    if (src_dims[i] < pads_begin[i] || src_dims[i] < pads_end[i])
                        THROW_IE_EXCEPTION << layer->name << " Incorrect pads_begin or pads_end for 'symmetric' pad mode";
                }
            } else {
                THROW_IE_EXCEPTION << layer->name
                                   << " Incorrect pad_mode. Only constants|edge|reflect|symmetric modes are supported!";
            }

            if (padMode == CONSTANT)
                pad_value = layer->GetParamAsFloat("pad_value", 0.f);

            srcStrides = layer->insData[0].lock()->getTensorDesc().getBlockingDesc().getStrides();
            dstStrides = layer->outData[0]->getTensorDesc().getBlockingDesc().getStrides();
            work_amount = dst_dims[0] * dstStrides[0];
            for (size_t i = 0; i < src_dims.size(); i++)
                src_o_dms.push_back(src_dims[i] + pads_begin[i]);

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

        switch (padMode) {
            case CONSTANT:
                pad_constant(src_data, dst_data);
                break;
            case EDGE:
                pad_edge(src_data, dst_data);
                break;
            case REFLECT:
                pad_reflect(src_data, dst_data);
                break;
            case SYMMETRIC:
                pad_symmetric(src_data, dst_data);
                break;
            default:
                return GENERAL_ERROR;
        }
        return OK;
    }

private:
    enum PadMode {
        CONSTANT = 0,
        EDGE = 1,
        REFLECT = 2,
        SYMMETRIC = 3
    };

    void pad_constant(const float *src_data, float* dst_data);
    void pad_edge(const float *src_data, float* dst_data);
    void pad_reflect(const float *src_data, float* dst_data);
    void pad_symmetric(const float *src_data, float* dst_data);

    PadMode padMode = CONSTANT;
    float pad_value = 0.f;
    SizeVector src_dims;
    SizeVector dst_dims;
    std::vector<unsigned int> pads_begin;
    SizeVector src_o_dms;
    SizeVector srcStrides;
    SizeVector dstStrides;
    size_t work_amount;
};


inline size_t parallel_init(size_t start, size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = start % dims[j];
        start = start / dims[j];
    }
    return start;
}

inline void parallel_step(size_t size, std::vector<size_t> &counters, std::vector<size_t> &dims) {
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = (counters[j] + 1) % dims[j];
        if (counters[j] != 0)
            return;
    }
}

void PadImpl::pad_constant(const float *src_data, float* dst_data) {
    int offset = 0;
    for (size_t i = 0; i < srcStrides.size(); ++i)
        offset += pads_begin[i] * srcStrides[i];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, dst_dims.size(), counters, dst_dims);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int srcIdx = 1;
            int dstIdx = 0;
            for (size_t i = 0; i < dstStrides.size(); ++i)
                dstIdx += counters[i] * dstStrides[i];

            for (size_t i = 0; i < counters.size(); ++i) {
                if (counters[i] < pads_begin[i] || counters[i] >= src_o_dms[i]) {
                    dst_data[dstIdx] = pad_value;
                    srcIdx = 0;
                    break;
                }
            }
            if (srcIdx) {
                int srcIdx = 0;
                for (size_t i = 0; i < srcStrides.size(); ++i)
                    srcIdx += counters[i] * srcStrides[i];
                dst_data[dstIdx] = src_data[srcIdx - offset];
            }
            parallel_step(dst_dims.size(), counters, dst_dims);
        }
    });
}

void PadImpl::pad_edge(const float *src_data, float* dst_data) {
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, dst_dims.size(), counters, dst_dims);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int srcIdx = 0;
            int dstIdx = 0;
            for (size_t i = 0; i < dstStrides.size(); ++i)
                dstIdx += counters[i] * dstStrides[i];

            for (size_t i = 0; i < srcStrides.size(); ++i) {
                int idx = (counters[i] < pads_begin[i]) ? 0 :
                    ((counters[i] >= src_o_dms[i]) ? (src_dims[i] - 1) : (counters[i] - pads_begin[i]));
                srcIdx += idx * srcStrides[i];
            }

            dst_data[dstIdx] = src_data[srcIdx];
            parallel_step(dst_dims.size(), counters, dst_dims);
        }
    });
}

void PadImpl::pad_reflect(const float *src_data, float* dst_data) {
    SizeVector src_2;
    for (size_t i = 0; i < src_dims.size(); i++)
        src_2.push_back(src_dims[i] + src_o_dms[i] - 2);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, dst_dims.size(), counters, dst_dims);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int srcIdx = 0;
            int dstIdx = 0;
            for (size_t i = 0; i < dstStrides.size(); ++i)
                dstIdx += counters[i] * dstStrides[i];

            for (size_t i = 0; i < srcStrides.size(); ++i) {
                int idx = (counters[i] < pads_begin[i]) ? (pads_begin[i] - counters[i]) :
                    ((counters[i] >= src_o_dms[i]) ? (src_2[i] - counters[i]) : (counters[i] - pads_begin[i]));
                srcIdx += idx * srcStrides[i];
            }

            dst_data[dstIdx] = src_data[srcIdx];
            parallel_step(dst_dims.size(), counters, dst_dims);
        }
    });
}

void PadImpl::pad_symmetric(const float *src_data, float* dst_data) {
    SizeVector src_2;
    for (size_t i = 0; i < src_dims.size(); i++)
        src_2.push_back(src_dims[i] + src_o_dms[i] - 1);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount, nthr, ithr, start, end);

        parallel_init(start, dst_dims.size(), counters, dst_dims);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int srcIdx = 0;
            int dstIdx = 0;
            for (size_t i = 0; i < dstStrides.size(); ++i)
                dstIdx += counters[i] * dstStrides[i];

            for (size_t i = 0; i < srcStrides.size(); ++i) {
                int idx = (counters[i] < pads_begin[i]) ? (pads_begin[i] - 1 - counters[i]) :
                    ((counters[i] >= src_o_dms[i]) ? (src_2[i] - counters[i]) : (counters[i] - pads_begin[i]));
                srcIdx += idx * srcStrides[i];
            }

            dst_data[dstIdx] = src_data[srcIdx];
            parallel_step(dst_dims.size(), counters, dst_dims);
        }
    });
}

REG_FACTORY_FOR(ImplFactory<PadImpl>, Pad);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
