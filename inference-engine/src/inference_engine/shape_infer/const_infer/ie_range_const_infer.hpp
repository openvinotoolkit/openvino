// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ie_layers.h>
#include <ie_memcpy.h>
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for Fill layer
 */
class RangeConstInfer : public ConstInferImpl {
public:
    explicit RangeConstInfer(const std::string& type) : ConstInferImpl(type) {}

    template<typename data_t>
    void range(data_t start, data_t limit, data_t delta, const Blob::Ptr& output) {
        size_t dst_size = (output->getTensorDesc().getDims())[0];
        data_t* dst_data = output->cbuffer().as<data_t*>() +
                           output->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t work_amount_dst = std::floor(std::abs((limit - start) / delta));
        if (work_amount_dst != dst_size)
            THROW_IE_EXCEPTION << "Range indexes exceeds data tensor dimension";

        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t iwork = 0, end = 0;
            splitter(work_amount_dst, nthr, ithr, iwork, end);
            data_t dst_value = start + iwork * delta;

            for (; iwork < end; ++iwork, dst_value += delta) {
                dst_data[iwork] = dst_value;
            }
        });
    }

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        const size_t RANGE_START = 0;
        const size_t RANGE_LIMIT = 1;
        const size_t RANGE_DELTA = 2;
        if (inData.empty() || outData.empty())
            THROW_IE_EXCEPTION << " Incorrect number of input/output edges!";

        if (inData.size() != 3)
            THROW_IE_EXCEPTION << " Incorrect number of input edges!";

        SizeVector start_dims = inData[RANGE_START]->getTensorDesc().getDims();
        if (start_dims.size() > 1)
            THROW_IE_EXCEPTION << " Start scalar should have 1 dimension";

        SizeVector limit_dims = inData[RANGE_LIMIT]->getTensorDesc().getDims();
        if (limit_dims.size() > 1)
            THROW_IE_EXCEPTION << " Limit scalar should have 1 dimension";

        SizeVector delta_dims = inData[RANGE_DELTA]->getTensorDesc().getDims();
        if (delta_dims.size() > 1)
            THROW_IE_EXCEPTION << " Delta scalar should have 1 dimension";

        SizeVector dst_dims = outData[0]->getTensorDesc().getDims();
        if (dst_dims.size() > 1)
            THROW_IE_EXCEPTION << " Output vector should have 1 dimension";

        if (!(inData[RANGE_START]->getTensorDesc().getPrecision() == Precision::I32 &&
              inData[RANGE_LIMIT]->getTensorDesc().getPrecision() == Precision::I32 &&
              inData[RANGE_DELTA]->getTensorDesc().getPrecision() == Precision::I32 &&
              outData[0]->getTensorDesc().getPrecision() == Precision::I32) &&
            !(inData[RANGE_START]->getTensorDesc().getPrecision() == Precision::FP32 &&
              inData[RANGE_LIMIT]->getTensorDesc().getPrecision() == Precision::FP32 &&
              inData[RANGE_DELTA]->getTensorDesc().getPrecision() == Precision::FP32 &&
              outData[0]->getTensorDesc().getPrecision() == Precision::FP32)) {
            THROW_IE_EXCEPTION <<
                               " 'Start', 'Limit', 'Delta' input scalars and output tensor should have same precision"
                               <<
                               "and only FP32 and I32 are supported!";
        }

        StatusCode retcode = OK;
        switch (outData[0]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                range((inData[RANGE_START]->cbuffer().as<float*>() +
                       inData[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                      (inData[RANGE_LIMIT]->cbuffer().as<float*>() +
                       inData[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                      (inData[RANGE_DELTA]->cbuffer().as<float*>() +
                       inData[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outData[0]);
            }
                break;
            case Precision::I32: {
                range((inData[RANGE_START]->cbuffer().as<int32_t*>() +
                       inData[RANGE_START]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                      (inData[RANGE_LIMIT]->cbuffer().as<int32_t*>() +
                       inData[RANGE_LIMIT]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0],
                      (inData[RANGE_DELTA]->cbuffer().as<int32_t*>() +
                       inData[RANGE_DELTA]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0], outData[0]);
            }
                break;
            default:
                THROW_IE_EXCEPTION << "Incorrect output precision. Only FP32 and I32 are supported!";
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
