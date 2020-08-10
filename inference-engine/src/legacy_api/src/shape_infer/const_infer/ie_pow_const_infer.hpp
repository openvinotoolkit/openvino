// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <legacy/ie_layers.h>
#include <precision_utils.h>

#include <cmath>
#include <ie_precision.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "broadcast_offset.hpp"

namespace InferenceEngine {
namespace ShapeInfer {
class PowConstInfer : public ConstInferImpl {
public:
    explicit PowConstInfer(const std::string& type): ConstInferImpl(type) {}

    struct fp16tofp32 {
        inline float operator()(ie_fp16 value) {
            return static_cast<float>(PrecisionUtils::f16tof32(value));
        }
    };

    struct fp32tofp16 {
        inline ie_fp16 operator()(float value) {
            return static_cast<float>(PrecisionUtils::f32tof16(value));
        }
    };

    template <typename dataType>
    struct noConversion {
        inline dataType operator()(dataType value) {
            return value;
        }
    };

    template <typename inDatatype1, typename inDatatype2, typename outDatatype, class ConversionInData1,
              class ConversionInData2, class ConversionOutData>
    void pow(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
             const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) {
        auto* firstBlobBuffer = inData[0]->cbuffer().as<inDatatype1*>();
        auto* secondBlobBuffer = inData[1]->cbuffer().as<inDatatype2*>();
        if (!firstBlobBuffer || !secondBlobBuffer) {
            THROW_IE_EXCEPTION << "empty input data";
        }

        auto outBlob = *outData.begin();
        auto* outBuffer = outBlob->buffer().as<outDatatype*>();
        if (!outBuffer) THROW_IE_EXCEPTION << "empty output data";

        BroadcastOffset outOff(outBlob->getTensorDesc().getDims(), outBlob->getTensorDesc().getDims());
        BroadcastOffset inOff1(inData[0]->getTensorDesc().getDims(), outBlob->getTensorDesc().getDims());
        BroadcastOffset inOff2(inData[1]->getTensorDesc().getDims(), outBlob->getTensorDesc().getDims());
        for (size_t i = 0; i < outBlob->size(); i++) {
            SizeVector offsetDims = outOff.offset_dims(i);
            outBuffer[outOff.offset(offsetDims)] =
                ConversionOutData()(std::pow(ConversionInData1()(firstBlobBuffer[inOff1.offset(offsetDims)]),
                                             ConversionInData2()(secondBlobBuffer[inOff2.offset(offsetDims)])));
        }
    }

    void inferImpl(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
                   const std::map<std::string, Blob::Ptr>& blobs, std::vector<Blob::Ptr>& outData) override {
        size_t numInputs = inData.size();
        if (inData.size() != 2)
            THROW_IE_EXCEPTION << "Unsupported number of inputs: " << numInputs << ". 2 inputs is supported";

        auto compare =
            getPrecisionMask(inData[0]->getTensorDesc().getPrecision(), inData[1]->getTensorDesc().getPrecision(),
                             outData[0]->getTensorDesc().getPrecision());
        switch (compare) {
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::FP32):
            pow<float, float, float, noConversion<float>, noConversion<float>, noConversion<float>>(inData, params,
                                                                                                    blobs, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::FP32):
            pow<int32_t, int32_t, float, noConversion<int32_t>, noConversion<int32_t>, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP16, Precision::FP16):
            pow<ie_fp16, ie_fp16, ie_fp16, noConversion<ie_fp16>, noConversion<ie_fp16>, noConversion<ie_fp16>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::FP16):
            pow<int32_t, int32_t, float, noConversion<int32_t>, noConversion<int32_t>, fp32tofp16>(inData, params,
                                                                                                   blobs, outData);
            break;
        default:
            THROW_IE_EXCEPTION << "Not supported data type in port 0";
        }
    }
};
}  // namespace ShapeInfer
}  // namespace InferenceEngine