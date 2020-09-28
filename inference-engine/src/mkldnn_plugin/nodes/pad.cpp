// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

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
                pad_value = layer->GetParamAsFloat("pad_value", 0.f);
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

            srcStrides = layer->insData[0].lock()->getTensorDesc().getBlockingDesc().getStrides();
            dstStrides = layer->outData[0]->getTensorDesc().getBlockingDesc().getStrides();
            work_amount = dst_dims[0] * dstStrides[0];
            pad_dims.resize(pads_begin.size());
            for (size_t i = 0; i < src_dims.size(); i++) {
                src_o_dms.push_back(src_dims[i] + pads_begin[i]);
                pad_points_num += pads_begin[i] + pads_end[i];
                pad_dims[i] = pads_begin[i] + pads_end[i];
            }

            addConfig(layer, { DataConfigurator(ConfLayout::PLN, Precision::FP32) }, { DataConfigurator(ConfLayout::PLN, Precision::FP32) });
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
    size_t pad_points_num = 0;
    SizeVector pad_dims;
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
    //  Vectorized copy
    size_t dims_size_1 = dst_dims.size() - 1;
    size_t inputSV = src_dims[dims_size_1];
    size_t work_amount_src = srcStrides[0] * src_dims[0] / src_dims[dims_size_1];


    int offset = 0;
    for (size_t i = 0; i < srcStrides.size(); ++i)
        offset += pads_begin[i] * dstStrides[i];
    std::fill_n(dst_data, offset, pad_value);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        //const int ithr = 0, const int nthr = 1;
        size_t start = 0, end = 0;
        SizeVector counters(dims_size_1, 0);
        splitter(work_amount_src, nthr, ithr, start, end);
        SizeVector counters1(dims_size_1, 0);

        parallel_init(start, dims_size_1, counters, src_dims);
        parallel_init(start, dims_size_1, counters1, src_dims);
        parallel_step(dims_size_1, counters1, src_dims);
        int src_idx = 0;
        for (size_t i = 0; i < dims_size_1; ++i)
            src_idx += counters[i] * srcStrides[i];
        int dst_idx = pads_begin[dims_size_1];
        for (size_t i = 0; i < dims_size_1; ++i)
            dst_idx += (pads_begin[i] + counters[i]) * dstStrides[i];
        int dst_idx1 = pads_begin[dims_size_1];
        for (size_t i = 0; i < dims_size_1; ++i)
            dst_idx1 += (pads_begin[i] + counters1[i]) * dstStrides[i];
        if (dst_idx1 <= dst_idx) dst_idx1 = dstStrides[0] * dst_dims[0];

        for (size_t iwork = start; iwork < end; ++iwork, src_idx += inputSV) {
            cpu_memcpy(&dst_data[dst_idx], &src_data[src_idx], sizeof(float) * inputSV);
            std::fill_n(&dst_data[dst_idx + inputSV], dst_idx1 - dst_idx - inputSV, pad_value);

            //parallel_step(dims_size_1, counters, src_dims);
            for (int j = dims_size_1 - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % src_dims[j];
                if (counters[j] != 0) {
                    dst_idx += dstStrides[j];
                    break;
                } else {
                    dst_idx = pads_begin[dims_size_1];
                    for (size_t i = 0; i < dims_size_1; ++i)
                        dst_idx += (pads_begin[i] + counters[i]) * dstStrides[i];
                }
            }
            //parallel_step(dims_size_1, counters1, src_dims);
            for (int j = dims_size_1 - 1; j >= 0; j--) {
                counters1[j] = (counters1[j] + 1) % src_dims[j];
                if (counters1[j] != 0) {
                    dst_idx1 += dstStrides[j];
                    break;
                } else {
                    dst_idx1 = pads_begin[dims_size_1];
                    for (size_t i = 0; i < dims_size_1; ++i)
                        dst_idx1 += (pads_begin[i] + counters1[i]) * dstStrides[i];
                }
            }
            if (dst_idx1 <= dst_idx) dst_idx1 = dstStrides[0] * dst_dims[0];
        }
    });
}

void PadImpl::pad_edge(const float *src_data, float* dst_data) {
    //  Vectorized copy
    size_t dims_size_1 = dst_dims.size() - 1;
    size_t inputSV = dst_dims[dims_size_1];
    size_t work_amount_dst = dstStrides[0] * dst_dims[0] / dst_dims[dims_size_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        //const int ithr = 0, const int nthr = 1;
        size_t start = 0, end = 0;
        SizeVector counters(dims_size_1, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);

        parallel_init(start, dims_size_1, counters, dst_dims);
        int dst_idx = 0;
        for (size_t i = 0; i < dims_size_1; ++i)
            dst_idx += counters[i] * dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dims_size_1; ++i) {
                int idx = (counters[i] < pads_begin[i]) ? 0 :
                    ((counters[i] >= src_o_dms[i]) ? (src_dims[i] - 1) : (counters[i] - pads_begin[i]));
                src_idx += idx * srcStrides[i];
            }

            std::fill_n(&dst_data[dst_idx], pads_begin[dims_size_1], src_data[src_idx]);
            cpu_memcpy(&dst_data[dst_idx + pads_begin[dims_size_1]], &src_data[src_idx], sizeof(float) * src_dims[dims_size_1]);
            std::fill_n(&dst_data[dst_idx + src_o_dms[dims_size_1]], dst_dims[dims_size_1] - src_o_dms[dims_size_1], src_data[src_idx+src_dims[dims_size_1]-1]);

            parallel_step(dims_size_1, counters, dst_dims);
        }
    });
}

void PadImpl::pad_reflect(const float *src_data, float* dst_data) {
    SizeVector src_2;
    for (size_t i = 0; i < src_dims.size(); i++)
        src_2.push_back(src_dims[i] + src_o_dms[i] - 2);

    //  Vectorized copy
    size_t dims_size_1 = dst_dims.size() - 1;
    size_t inputSV = dst_dims[dims_size_1];
    size_t work_amount_dst = dstStrides[0] * dst_dims[0] / dst_dims[dims_size_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        //const int ithr = 0, const int nthr = 1;
        size_t start = 0, end = 0;
        SizeVector counters(dims_size_1, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);

        parallel_init(start, dims_size_1, counters, dst_dims);
        int dst_idx = 0;
        for (size_t i = 0; i < dims_size_1; ++i)
            dst_idx += counters[i] * dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dims_size_1; ++i) {
                int idx = (counters[i] < pads_begin[i]) ? (pads_begin[i] - counters[i]) :
                    ((counters[i] >= src_o_dms[i]) ? (src_2[i] - counters[i]) : (counters[i] - pads_begin[i]));
                src_idx += idx * srcStrides[i];
            }

            for (size_t i = 0; i < pads_begin[dims_size_1]; ++i) {
                dst_data[dst_idx + i] = src_data[src_idx + pads_begin[dims_size_1] - i];
            }
            cpu_memcpy(&dst_data[dst_idx + pads_begin[dims_size_1]], &src_data[src_idx], sizeof(float) * src_dims[dims_size_1]);
            for (size_t i = src_o_dms[dims_size_1]; i < dst_dims[dims_size_1]; ++i) {
                dst_data[dst_idx + i] = src_data[src_idx + src_2[dims_size_1] - i];
            }

            parallel_step(dims_size_1, counters, dst_dims);
        }
    });
}

void PadImpl::pad_symmetric(const float *src_data, float* dst_data) {
    SizeVector src_2;
    for (size_t i = 0; i < src_dims.size(); i++)
        src_2.push_back(src_dims[i] + src_o_dms[i] - 1);

    //  Vectorized copy
    size_t dims_size_1 = dst_dims.size() - 1;
    size_t inputSV = dst_dims[dims_size_1];
    size_t work_amount_dst = dstStrides[0] * dst_dims[0] / dst_dims[dims_size_1];

    parallel_nt(0, [&](const int ithr, const int nthr) {
        //const int ithr = 0, const int nthr = 1;
        size_t start = 0, end = 0;
        SizeVector counters(dims_size_1, 0);
        splitter(work_amount_dst, nthr, ithr, start, end);

        parallel_init(start, dims_size_1, counters, dst_dims);
        int dst_idx = 0;
        for (size_t i = 0; i < dims_size_1; ++i)
            dst_idx += counters[i] * dstStrides[i];

        for (size_t iwork = start; iwork < end; ++iwork, dst_idx += inputSV) {
            int src_idx = 0;
            for (size_t i = 0; i < dims_size_1; ++i) {
                int idx = (counters[i] < pads_begin[i]) ? (pads_begin[i] - 1 - counters[i]) :
                    ((counters[i] >= src_o_dms[i]) ? (src_2[i] - counters[i]) : (counters[i] - pads_begin[i]));
                src_idx += idx * srcStrides[i];
            }

            for (size_t i = 0; i < pads_begin[dims_size_1]; ++i) {
                dst_data[dst_idx + i] = src_data[src_idx + pads_begin[dims_size_1] -1 - i];
            }
            cpu_memcpy(&dst_data[dst_idx + pads_begin[dims_size_1]], &src_data[src_idx], sizeof(float) * src_dims[dims_size_1]);
            for (size_t i = src_o_dms[dims_size_1]; i < dst_dims[dims_size_1]; ++i) {
                dst_data[dst_idx + i] = src_data[src_idx + src_2[dims_size_1] - i];
            }

            parallel_step(dims_size_1, counters, dst_dims);
        }
    });
}

REG_FACTORY_FOR(PadImpl, Pad);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
