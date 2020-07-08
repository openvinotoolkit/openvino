// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "ie_parallel.hpp"
#include "common/simple_copy.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

inline void align_dims(SizeVector &dims_aligned, const SizeVector &dims) {
    for (int i = 0; i < dims.size(); i++) {
        dims_aligned[dims_aligned.size() - 1 - i] = dims[dims.size() - 1 - i];
    }
}

inline const int get_offset(SizeVector &strides, int i0, int i1, int i2, int i3, int i4) {
    return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3] + i4 * strides[4];
}

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

            LayerConfig config;
            DataConfig dataConfig, shapeConfig;
            Precision dataPrecision = layer->outData[0]->getTensorDesc().getPrecision();
            const SizeVector& data_dims = layer->insData[BROADCAST_INPUT].lock()->getTensorDesc().getDims();
            dataConfig.desc = TensorDesc(dataPrecision, data_dims,
                                         layer->insData[BROADCAST_INPUT].lock()->getTensorDesc().getLayout());
            config.inConfs.push_back(dataConfig);
            shapeConfig.desc = TensorDesc(layer->insData[BROADCAST_SHAPE].lock()->getTensorDesc().getPrecision(),
                                          shape_dims, TensorDesc::getLayoutByDims(shape_dims));
            config.inConfs.push_back(shapeConfig);

            DataConfig outConfig;
            const SizeVector& out_dims = layer->outData[0]->getTensorDesc().getDims();
            outConfig.desc = TensorDesc(dataPrecision, out_dims, layer->outData[0]->getTensorDesc().getLayout());
            config.outConfs.push_back(outConfig);
            config.dynBatchSupport = false;
            confs.push_back(config);
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        SizeVector src_dims = inputs[BROADCAST_INPUT]->getTensorDesc().getDims();
        SizeVector dst_dims = outputs[0]->getTensorDesc().getDims();
        SizeVector srcStrides = inputs[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getStrides();
        SizeVector dstStrides = outputs[0]->getTensorDesc().getBlockingDesc().getStrides();
        size_t data_size = inputs[BROADCAST_INPUT]->getTensorDesc().getPrecision().size();

        const uint8_t *src_data = inputs[BROADCAST_INPUT]->cbuffer().as<const uint8_t *>() +
                                  inputs[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t* dst_data = outputs[0]->cbuffer().as<uint8_t *>() +
                            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

        const int maxDimsForOptimizedCase = 6;
        if (dst_dims.size() <= maxDimsForOptimizedCase) {
            SizeVector src_aligned(maxDimsForOptimizedCase, 1), dst_aligned(maxDimsForOptimizedCase, 1);
            align_dims(src_aligned, src_dims);
            align_dims(dst_aligned, dst_dims);

            SizeVector srcStrides_aligned(maxDimsForOptimizedCase, srcStrides[0]), dstStrides_aligned(maxDimsForOptimizedCase, dstStrides[0]);
            align_dims(srcStrides_aligned, srcStrides);
            align_dims(dstStrides_aligned, dstStrides);

            for (int i = 0; i < maxDimsForOptimizedCase; i++) {
                if (src_aligned[i] != dst_aligned[i]) {
                    srcStrides_aligned[i] = 0;
                }
            }

            if (dst_dims[dst_dims.size() - 1] == src_dims[src_dims.size() - 1]) {
                parallel_for5d(dst_aligned[0], dst_aligned[1], dst_aligned[2], dst_aligned[3], dst_aligned[4],
                        [&](int i0, int i1, int i2, int i3, int i4) {
                    const uint8_t *src_data_2 = src_data + get_offset(srcStrides_aligned, i0, i1, i2, i3, i4) * data_size;
                    uint8_t *dst_data_2 = dst_data + get_offset(dstStrides_aligned, i0, i1, i2, i3, i4) * data_size;
                    simple_copy(dst_data_2, data_size * dst_aligned[5], src_data_2, data_size * dst_aligned[5]);
                });
            } else {
                parallel_for5d(dst_aligned[0], dst_aligned[1], dst_aligned[2], dst_aligned[3], dst_aligned[4],
                        [&](int i0, int i1, int i2, int i3, int i4) {
                    const uint8_t *src_data_2 = src_data + get_offset(srcStrides_aligned, i0, i1, i2, i3, i4) * data_size;
                    uint8_t *dst_data_2 = dst_data + get_offset(dstStrides_aligned, i0, i1, i2, i3, i4) * data_size;
                    for (int i = 0; i < dst_aligned[5]; i++) {
                        simple_copy(dst_data_2 + i * data_size, data_size, src_data_2, data_size);
                    }
                });
            }
        } else {
            size_t shape_size = (inputs[BROADCAST_SHAPE]->getTensorDesc().getDims())[0];

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
        }

        return OK;
    }

private:
    const size_t BROADCAST_INPUT = 0;
    const size_t BROADCAST_SHAPE = 1;
};

REG_FACTORY_FOR(BroadcastImpl, Broadcast);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
