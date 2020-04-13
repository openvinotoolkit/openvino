// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include <ie_layers.h>
#include <precision_utils.h>

#include <ie_precision.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "broadcast_offset.hpp"
#include "ie_const_infer_impl.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

/**
 *@brief Implementation of Const inference for TBD layer
 *
 * Table of output data type value with given input parameters
 *
 *
 *              U8       I32        I64        FP16        FP32
 *     =============================================================
 *     U8   ==  U8       I32        I64        FP16        FP32
 *          ==
 *     I32  ==  I32      I32        I64        FP32        FP32
 *          ==
 *     I64  ==  I64      I64        I64        FP32        FP32
 *          ==
 *     FP16 ==  FP16     FP32       FP32       FP16        FP32
 *          ==
 *     FP32 ==  FP32     FP32       FP32       FP32        FP32
 *
 *     There is a special case with FP16 precision. Convert input data to FP32 and multiply. After that
 *     convert output data to FP16, if both of input parameters have FP16 precision or one - FP16 and another - U8.
 */

class MulConstInfer : public ConstInferImpl {
public:
    explicit MulConstInfer(const std::string& type): ConstInferImpl(type) {}

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
    void mul(const std::vector<Blob::CPtr>& inData, const std::map<std::string, std::string>& params,
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
                ConversionOutData()(ConversionInData1()(firstBlobBuffer[inOff1.offset(offsetDims)]) *
                                    ConversionInData2()(secondBlobBuffer[inOff2.offset(offsetDims)]));
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
        case getPrecisionMask(Precision::U8, Precision::U8, Precision::U8):
            mul<uint8_t, uint8_t, uint8_t, noConversion<uint8_t>, noConversion<uint8_t>, noConversion<uint8_t>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U8, Precision::I32, Precision::I32):
            mul<uint8_t, int, int, noConversion<uint8_t>, noConversion<int>, noConversion<int>>(inData, params, blobs,
                                                                                                outData);
            break;
        case getPrecisionMask(Precision::U8, Precision::I64, Precision::I64):
            mul<uint8_t, long long int, long long int, noConversion<uint8_t>, noConversion<long long int>,
                noConversion<long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U8, Precision::U64, Precision::U64):
            mul<uint8_t, unsigned long long int, unsigned long long int, noConversion<uint8_t>,
                noConversion<unsigned long long int>, noConversion<unsigned long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U8, Precision::FP16, Precision::FP16):
            mul<uint8_t, ie_fp16, ie_fp16, noConversion<uint8_t>, fp16tofp32, fp32tofp16>(inData, params, blobs,
                                                                                          outData);
            break;
        case getPrecisionMask(Precision::U8, Precision::FP32, Precision::FP32):
            mul<uint8_t, float, float, noConversion<uint8_t>, noConversion<float>, noConversion<float>>(inData, params,
                                                                                                        blobs, outData);
            break;

        case getPrecisionMask(Precision::I32, Precision::U8, Precision::I32):
            mul<int, uint8_t, int, noConversion<int>, noConversion<uint8_t>, noConversion<int>>(inData, params, blobs,
                                                                                                outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I32):
            mul<int, int, int, noConversion<int>, noConversion<int>, noConversion<int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I64):
            mul<int, long long int, long long int, noConversion<int>, noConversion<long long int>,
                noConversion<long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::U64, Precision::U64):
            mul<int, unsigned long long int, unsigned long long int, noConversion<int>,
                noConversion<unsigned long long int>, noConversion<unsigned long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::FP16, Precision::FP32):
            mul<int, ie_fp16, float, noConversion<int>, fp16tofp32, noConversion<float>>(inData, params, blobs,
                                                                                         outData);
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::FP32):
            mul<int, float, float, noConversion<int>, noConversion<float>, noConversion<float>>(inData, params, blobs,
                                                                                                outData);
            break;

        case getPrecisionMask(Precision::I64, Precision::U8, Precision::I64):
            mul<long long int, uint8_t, long long int, noConversion<long long int>, noConversion<uint8_t>,
                noConversion<long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I64):
            mul<long long int, int, long long int, noConversion<long long int>, noConversion<int>,
                noConversion<long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I64):
            mul<long long int, long long int, long long int, noConversion<long long int>, noConversion<long long int>,
                noConversion<long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I64, Precision::FP16, Precision::FP32):
            mul<long long int, ie_fp16, float, noConversion<long long int>, fp16tofp32, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::FP32):
            mul<long long int, float, float, noConversion<long long int>, noConversion<float>, noConversion<float>>(
                inData, params, blobs, outData);
            break;

        case getPrecisionMask(Precision::U64, Precision::U8, Precision::U64):
            mul<unsigned long long int, uint8_t, unsigned long long int, noConversion<unsigned long long int>,
                noConversion<uint8_t>, noConversion<unsigned long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U64, Precision::I32, Precision::U64):
            mul<unsigned long long int, int, unsigned long long int, noConversion<unsigned long long int>,
                noConversion<int>, noConversion<unsigned long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U64, Precision::U64, Precision::U64):
            mul<unsigned long long int, unsigned long long int, unsigned long long int,
                noConversion<unsigned long long int>, noConversion<unsigned long long int>,
                noConversion<unsigned long long int>>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U64, Precision::FP16, Precision::FP32):
            mul<unsigned long long int, ie_fp16, float, noConversion<unsigned long long int>, fp16tofp32, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::U64, Precision::FP32, Precision::FP32):
            mul<unsigned long long int, float, float, noConversion<unsigned long long int>, noConversion<float>, noConversion<float>>(
                inData, params, blobs, outData);
            break;

        case getPrecisionMask(Precision::FP16, Precision::U8, Precision::FP16):
            mul<ie_fp16, uint8_t, ie_fp16, fp16tofp32, noConversion<uint8_t>, fp32tofp16>(inData, params, blobs,
                                                                                          outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::I32, Precision::FP32):
            mul<ie_fp16, int, float, fp16tofp32, noConversion<int>, noConversion<float>>(inData, params, blobs,
                                                                                         outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::I64, Precision::FP32):
            mul<ie_fp16, long long int, float, fp16tofp32, noConversion<long long int>, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::U64, Precision::FP32):
            mul<ie_fp16, unsigned long long int, float, fp16tofp32, noConversion<unsigned long long int>, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP16, Precision::FP16):
            mul<ie_fp16, ie_fp16, ie_fp16, fp16tofp32, fp16tofp32, fp32tofp16>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP32, Precision::FP32):
            mul<ie_fp16, float, float, fp16tofp32, noConversion<float>, noConversion<float>>(inData, params, blobs,
                                                                                             outData);
            break;
        case getPrecisionMask(Precision::FP16, Precision::FP32, Precision::FP16):
            mul<ie_fp16, float, ie_fp16, fp16tofp32, noConversion<float>, fp32tofp16>(inData, params, blobs, outData);
            break;

        case getPrecisionMask(Precision::FP32, Precision::U8, Precision::FP32):
            mul<float, uint8_t, float, noConversion<float>, noConversion<uint8_t>, noConversion<float>>(inData, params,
                                                                                                        blobs, outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::FP32):
            mul<float, int, float, noConversion<float>, noConversion<int>, noConversion<float>>(inData, params, blobs,
                                                                                                outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::FP32):
            mul<float, long long int, float, noConversion<float>, noConversion<long long int>, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::U64, Precision::FP32):
            mul<float, unsigned long long int, float, noConversion<float>, noConversion<unsigned long long int>, noConversion<float>>(
                inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP16, Precision::FP32):
            mul<float, ie_fp16, float, noConversion<float>, fp16tofp32, noConversion<float>>(inData, params, blobs,
                                                                                             outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP16, Precision::FP16):
            mul<float, ie_fp16, ie_fp16, noConversion<float>, fp16tofp32, fp32tofp16>(inData, params, blobs, outData);
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::FP32):
            mul<float, float, float, noConversion<float>, noConversion<float>, noConversion<float>>(inData, params,
                                                                                                    blobs, outData);
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported precision!";
        }
    }
};
}  // namespace ShapeInfer
}  // namespace InferenceEngine
