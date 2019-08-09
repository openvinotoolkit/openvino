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
class FillConstInfer : public ConstInferImpl {
public:
    explicit FillConstInfer(const std::string& type) : ConstInferImpl(type) {}

    void inferImpl(const std::vector<Blob::CPtr>& inData,
                   const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs,
                   std::vector<Blob::Ptr>& outData) override {
        const size_t FILL_DIMS = 0;
        const size_t FILL_VALUE = 1;
        if (inData.empty() || outData.empty())
            THROW_IE_EXCEPTION << " Incorrect number of input/output edges!";

        if (inData.size() != 2)
            THROW_IE_EXCEPTION << " Incorrect number of input edges!";

        SizeVector dims = inData[FILL_DIMS]->getTensorDesc().getDims();
        if (dims.size() > 1)
            THROW_IE_EXCEPTION << " Fill dimensions vector should be 1 dimension";

        if (inData[FILL_DIMS]->getTensorDesc().getPrecision() != Precision::I32)
            THROW_IE_EXCEPTION << " Fill dimensions vector should be I32!";

        SizeVector value_dims = inData[FILL_VALUE]->getTensorDesc().getDims();
        if (value_dims.size() > 1)
            THROW_IE_EXCEPTION << " Value scalar should have 1 dimension";

        if (!(inData[FILL_VALUE]->getTensorDesc().getPrecision() == Precision::I32 &&
              outData[0]->getTensorDesc().getPrecision() == Precision::I32) &&
            !(inData[FILL_VALUE]->getTensorDesc().getPrecision() == Precision::FP32 &&
              outData[0]->getTensorDesc().getPrecision() == Precision::FP32)) {
            THROW_IE_EXCEPTION <<
                               " 'Value' input scalars and output tensor should have same precision and only FP32 and I32 are supported!";
        }

        int32_t* fill_dims = inData[FILL_DIMS]->cbuffer().as<int32_t*>() +
                             inData[FILL_DIMS]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        size_t fill_size = inData[FILL_DIMS]->getTensorDesc().getDims()[0];
        SizeVector dst_dims = outData[0]->getTensorDesc().getDims();

        if (dst_dims.size() != fill_size) {
            THROW_IE_EXCEPTION << "Output tensor dimension mismatch";
        }

        size_t work_amount_dst = 1;
        for (size_t i = 0; i < dst_dims.size(); i++) {
            work_amount_dst *= fill_dims[i];
            if (static_cast<int>(dst_dims[i]) != fill_dims[i]) {
                THROW_IE_EXCEPTION << "Output tensor dimension size mismatch";
            }
        }

        switch (outData[0]->getTensorDesc().getPrecision()) {
            case Precision::FP32: {
                float* dst_data = outData[0]->cbuffer().as<float*>() +
                                  outData[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                float value = (inData[FILL_VALUE]->cbuffer().as<float*>() +
                               inData[FILL_VALUE]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

                parallel_nt(0, [&](const int ithr, const int nthr) {
                    size_t start = 0, end = 0;
                    splitter(work_amount_dst, nthr, ithr, start, end);
                    std::fill_n(dst_data + start, end - start, value);
                });
            }
                break;
            case Precision::I32: {
                int32_t* dst_data = outData[0]->cbuffer().as<int32_t*>() +
                                    outData[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
                int32_t value = (inData[FILL_VALUE]->cbuffer().as<int32_t*>() +
                                 inData[FILL_VALUE]->getTensorDesc().getBlockingDesc().getOffsetPadding())[0];

                parallel_nt(0, [&](const int ithr, const int nthr) {
                    size_t start = 0, end = 0;
                    splitter(work_amount_dst, nthr, ithr, start, end);
                    std::fill_n(dst_data + start, end - start, value);
                });
            }
                break;
            default:
                THROW_IE_EXCEPTION << "Incorrect output precision. Only FP32 and I32 are supported!";
        }
    }
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
