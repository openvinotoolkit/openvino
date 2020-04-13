// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>

#include <cmath>
#include <ie_algorithm.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_const_infer_impl.hpp"
#include "ie_memcpy.h"
#include "ie_parallel.hpp"
#include "precision_utils.h"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Broadcast layer
 */
class BroadcastConstInfer : public ConstInferImpl {
private:
    const size_t BROADCAST_INPUT = 0;
    const size_t BROADCAST_SHAPE = 1;

public:
    explicit BroadcastConstInfer(const std::string& type): ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        LayerParams lp {};
        CNNLayer layer(lp);
        layer.params = params;

        if (outData.empty()) THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

        if (inData.size() != 2) THROW_IE_EXCEPTION << "Incorrect number of input edges!";

        if (inData[BROADCAST_SHAPE]->getTensorDesc().getDims().size() > 1)
            THROW_IE_EXCEPTION << "Shape vector should be 1 dimension";

        size_t data_size = inData[BROADCAST_INPUT]->getTensorDesc().getPrecision().size();
        size_t shape_size = (inData[BROADCAST_SHAPE]->getTensorDesc().getDims())[0];
        SizeVector dst_dims = outData[0]->getTensorDesc().getDims();
        SizeVector src_dims = inData[BROADCAST_INPUT]->getTensorDesc().getDims();

        if (!src_dims.size()) src_dims = SizeVector(1, 1);

        if (dst_dims.size() != shape_size) {
            THROW_IE_EXCEPTION << "Output tensor dimension mismatch";
        }

        if (src_dims.size() > dst_dims.size()) {
            THROW_IE_EXCEPTION << "Output tensor dimension is smaller then input tensor dimension";
        }

        InferenceEngine::SizeVector dstStrides = outData[0]->getTensorDesc().getBlockingDesc().getStrides();
        InferenceEngine::SizeVector srcStrides =
            inData[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getStrides();
        InferenceEngine::SizeVector src_aligned(dst_dims.size());
        InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
        if (!srcStrides.size()) srcStrides = SizeVector(1, 1);

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
        const uint8_t* src_data = inData[BROADCAST_INPUT]->cbuffer().as<const uint8_t*>() +
                                  inData[BROADCAST_INPUT]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        uint8_t* dst_data =
            outData[0]->cbuffer().as<uint8_t*>() + outData[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

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

                ie_memcpy(&dst_data[iwork], data_size, &src_data[src_idx * data_size], data_size);

                for (int j = dst_dims.size() - 1; j >= 0; j--) {
                    counters[j] = (counters[j] + 1) % dst_dims[j];
                    if (counters[j] != 0) break;
                }
            }
        });
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
