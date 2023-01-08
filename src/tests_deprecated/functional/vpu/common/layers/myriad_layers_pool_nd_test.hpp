// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_layers_tests.hpp"
#include "vpu_tests_config.hpp"
#include "vpu_case_common.hpp"
#include "precision_utils.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cmath>
#include <cstdlib>

#define DEBUG 0

using namespace InferenceEngine;

using InputShape  = std::vector<int>;
using KernelShape = std::vector<int>;
using Strides     = std::vector<int>;
using PadsBegin   = std::vector<int>;
using PadsEnd     = std::vector<int>;
using AutoPad       = std::string;
using PoolingMethod = std::string;
using RoundingType  = std::string;
using ExcludePad    = bool;

using PoolNDTestParams =
    std::tuple<
        InputShape,
        KernelShape,
        Strides,
        PadsBegin,
        PadsEnd,
        AutoPad,
        PoolingMethod,
        RoundingType,
        ExcludePad>;

class PoolNDTest: public myriadLayerTestBaseWithParam<PoolNDTestParams>
{
protected:

    void testPoolND() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        DISABLE_IF(!CheckMyriadX() && !CheckMA2085());

        //
        // Get test parameters
        //

        const auto& params = GetParam();

        const std::vector<int>  inputShape = std::get<0>(params);
        const std::vector<int> kernelShape = std::get<1>(params);
        const std::vector<int>     strides = std::get<2>(params);
        const std::vector<int>   padsBegin = std::get<3>(params);
        const std::vector<int>   padsEnd   = std::get<4>(params);
        const std::string          autoPad = std::get<5>(params);
        const std::string    poolingMethod = std::get<6>(params);
        const std::string     roundingType = std::get<7>(params);
        const bool              excludePad = std::get<8>(params);

        // Only support not-interleaved layouts: CHW, NCHW, NCDHW, ...
        const bool interleaved = false;

        const int     inputNDims =  inputShape.size();
        const int    kernelNDims = kernelShape.size();
        const int   stridesNDims =     strides.size();
        const int padsBeginNDims =   padsBegin.size();
        const int   padsEndNDims =     padsEnd.size();

        //
        // Verify test parameters
        //

        IE_ASSERT(inputNDims >= 3); // CHW, NCHW, NCDHW, ...

        const int channelsNDims = 1;
        const int batchNDims = inputNDims > 3; // 0 if CHW, 1 if NCHW etc
        IE_ASSERT(inputNDims == kernelNDims + channelsNDims + batchNDims);

        //
        // Assume dims order like {N, C, ..., H, W}
        // where {..., H, W} are spacial dimensions
        //

        const int channelsDim = batchNDims;
        const int spacialDimsBegin = channelsDim + 1;
        const int inputChannels = inputShape[channelsDim];

        IE_ASSERT(inputChannels > 0);

        IE_ASSERT(kernelNDims > 0);
        IE_ASSERT(kernelNDims == stridesNDims || stridesNDims == 0);

        IE_ASSERT(autoPad == "same_lower" ||
                  autoPad == "same_upper" ||
                  autoPad == "valid" ||
                  autoPad == "");

        if (autoPad == "") {
            IE_ASSERT(padsBeginNDims == kernelNDims);
            IE_ASSERT(padsEndNDims == kernelNDims);
        } else {
            IE_ASSERT(padsBeginNDims == 0);
            IE_ASSERT(padsEndNDims == 0);
        }

        IE_ASSERT(poolingMethod == "avg" ||
                  poolingMethod == "max");

        IE_ASSERT(roundingType == "floor" ||
                  roundingType == "ceil" ||
                  roundingType == "");

        //
        // Derive other parameters of layer
        //

        std::vector<int> outputShape(inputNDims);
        for (int i = 0; i < kernelNDims; i++) {
            int strides_i = stridesNDims ? strides[i] : 1;

            int remainder_i = inputShape[i + spacialDimsBegin] % strides_i;
            int pads_i = kernelShape[i] - (remainder_i? remainder_i: strides_i);

            int pads_begin_i, pads_end_i;
            if (autoPad == "") {
                pads_begin_i = padsBegin[i];
                pads_end_i   = padsEnd[i];
            } else if (autoPad == "valid") {
                pads_begin_i = 0;
                pads_end_i   = 0;
            } else if (autoPad == "same_lower") {
                pads_end_i   = pads_i / 2;           // floor(pads_i / 2.)
                pads_begin_i = pads_i - pads_end_i;  //  ceil(pads_i / 2.)
            } else if (autoPad == "same_upper") {
                pads_begin_i = pads_i / 2;
                pads_end_i   = pads_i - pads_begin_i;
            } else {
                IE_ASSERT(false); // this must never happen
            }

            outputShape[i + spacialDimsBegin] = (inputShape[i + spacialDimsBegin]
                                               + pads_begin_i + pads_end_i
                                               - kernelShape[i]
                                                ) / strides_i + 1;
        }
        outputShape[channelsDim] = inputChannels;
        if (batchNDims) {
            outputShape[0] = inputShape[0]; // copy batch size
        }

        //
        // Initialize data
        //

        TBlob<uint8_t>::Ptr inputBlob = createPlainTBlob(inputShape, Precision::FP16);
        TBlob<uint8_t>::Ptr outputBlob = createPlainTBlob(outputShape, Precision::FP16);

        inputBlob->allocate();
        outputBlob->allocate();

        int inputNum = getTotalNum(inputShape);
        uint8_t* inputBlobDataPtr = inputBlob->data();

        // HACK: Fulfill random data with Gaussian distribution! (not uniform)
        //
        // WHY: While uniform distribution is OK for reference implementation,
        //      hardware convolution on Myriad X uses tricky quantization that
        //      is not accurace enough if input is white-noise.
        //
        //      Such quantization adjusts to image's histogram, which Gaussian
        //      noise may simulate more-or-less adequately.
        #if 0
        fulfillUniformly(inputBlobDataPtr, inputNum, Precision::FP16, 0, 255);
        #else
        fulfillGaussian(inputBlobDataPtr, inputNum, Precision::FP16, 128, 32);
        #endif

        //
        // Initialize network
        //

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        std::string model = createModel(inputShape,
                                        outputShape,
                                        kernelShape,
                                        strides,
                                        padsBegin,
                                        padsEnd,
                                        autoPad,
                                        poolingMethod,
                                        roundingType,
                                        excludePad);
        #if DEBUG
        std::cout << "model:\n" << model << "\n";
        #endif

        ASSERT_NO_THROW(readNetwork(model));

        const CNNNetwork& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["pooling"]->setPrecision(Precision::FP16);

        //
        // Infer
        //

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, _config));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
        
        Blob::Ptr inputValuesBlob;
        ASSERT_NO_THROW(inputValuesBlob = _inferRequest.GetBlob("input"));
        
        void* inputValuesBlobDataPtr = inputValuesBlob->buffer();
        std::memcpy(inputValuesBlobDataPtr, inputBlobDataPtr, inputNum * sizeof(ie_fp16));

        ASSERT_NO_THROW(_inferRequest.Infer());
        
        Blob::Ptr outputValuesBlob;
        ASSERT_NO_THROW(outputValuesBlob = _inferRequest.GetBlob("pooling"));
        
        //
        // Check result
        //

        Blob::Ptr refValuesBlob = make_shared_blob<ie_fp16>(outputValuesBlob->getTensorDesc());
        refValuesBlob->allocate();

        const ie_fp16 *inputData = inputValuesBlob->cbuffer().as<ie_fp16*>();
              ie_fp16 *referenceData = refValuesBlob->buffer().as<ie_fp16*>();

        ref_poolnd(inputData,
                   referenceData,
                   inputShape,
                   outputShape,
                   kernelShape,
                   strides,
                   padsBegin,
                   padsEnd,
                   autoPad,
                   interleaved,
                   poolingMethod,
                   roundingType,
                   excludePad);

        float tolerance = std::pow(getTotalNum(kernelShape), 1.f/kernelNDims) / 1000;

        CompareCommonRelative(outputValuesBlob, refValuesBlob, tolerance);
    }

private:

    // Convolution ND reference for FP16
    static
    void ref_poolnd(const ie_fp16           input[],
                          ie_fp16           output[],
                    const std::vector<int>& inputShape,
                    const std::vector<int>& outputShape,
                    const std::vector<int>& kernelShape,
                    const std::vector<int>& strides,
                    const std::vector<int>& padsBegin,
                    const std::vector<int>& padsEnd,
                    const std::string       autoPad,
                    const bool              interleaved,
                    const std::string       poolingMethod,
                    const std::string       roundingType,
                    const bool              excludePad) {
        //
        // Convert input into fp32 format
        //

        int  inputNDims =  inputShape.size();

        int batchDim    = inputNDims > 3 ? 0 : -1;
        int channelsDim = interleaved ? inputNDims - 1 : batchDim + 1;

        int  inputChannels =  inputShape[channelsDim];
        int outputChannels = outputShape[channelsDim];
        IE_ASSERT(inputChannels == outputChannels);

        size_t  inputsNum = getTotalNum(inputShape);
        size_t outputsNum = getTotalNum(outputShape);

        std::unique_ptr<float[]>  inputF32(new float  [inputsNum]);
        std::unique_ptr<float[]> outputF32(new float [outputsNum]);

        copyF16ToF32(input, inputF32.get(), inputsNum);

        //
        // Execute reference convolution
        //

        ref_poolnd_common(inputF32.get(),
                          outputF32.get(),
                          inputShape,
                          outputShape,
                          kernelShape,
                          strides,
                          padsBegin,
                          padsEnd,
                          autoPad,
                          interleaved,
                          poolingMethod,
                          roundingType,
                          excludePad);

        //
        // Convert output to fp16
        //

        copyF32ToF16(outputF32.get(), output, outputsNum);
    }

    // Convolution ND: reference for FP32
    //
    // Assume dims order like {N, C, ..., H, W}
    // where {..., H, W} are spacial dimensions
    //
    // Either {N, ..., H, W, C} if interleaved
    //
    // TODO: move this code into "conv_ref.cpp"
    static
    void ref_poolnd_common(const float input[],
                                 float output[],
               const std::vector<int>& inputShape,
               const std::vector<int>& outputShape,
               const std::vector<int>& kernelShape,
               const std::vector<int>& strides,
               const std::vector<int>& padsBegin,
               const std::vector<int>& padsEnd,
               const std::string     & autoPad,
                           const bool  interleaved,
               const std::string     & poolingMethod,
               const std::string     & roundingType,
                           const bool  excludePad) {
        //
        // Verify parameters
        //

        const int     inputNDims =  inputShape.size();
        const int    outputNDims = outputShape.size();
        const int    kernelNDims = kernelShape.size();
        const int   stridesNDims =     strides.size();
        const int padsBeginNDims =   padsBegin.size();
        const int   padsEndNDims =     padsEnd.size();

        IE_ASSERT(inputNDims == outputNDims);
        IE_ASSERT(inputNDims >= 3); // CHW, NCHW, NCDHW, ...

        const int channelsNDims = 1;
        const int batchNDims = inputNDims > 3; // 0 if CHW, 1 if NCHW etc
        IE_ASSERT(inputNDims == kernelNDims + channelsNDims + batchNDims);

        const int channelsDim      = interleaved ? inputNDims - 1 : batchNDims;
        const int spacialDimsBegin = interleaved ? batchNDims     : channelsDim + 1;

        const int  inputChannels =  inputShape[channelsDim];
        const int outputChannels = outputShape[channelsDim];

        IE_ASSERT(inputChannels > 0);
        IE_ASSERT(inputChannels == outputChannels);

        IE_ASSERT(kernelNDims > 0);
        IE_ASSERT(kernelNDims == stridesNDims || stridesNDims == 0);

        IE_ASSERT(autoPad == "same_lower" ||
                  autoPad == "same_upper" ||
                  autoPad == "valid" ||
                  autoPad == "");

        if (autoPad == "") {
            IE_ASSERT(padsBeginNDims == kernelNDims);
            IE_ASSERT(padsEndNDims == kernelNDims);
        } else {
            IE_ASSERT(padsBeginNDims == 0);
            IE_ASSERT(padsEndNDims == 0);
        }

        IE_ASSERT(poolingMethod == "avg" || poolingMethod == "max");

        enum PoolingMethodEnum { Max = 1, Avg = 2 };
        int pooling_method = poolingMethod == "avg" ? Avg : Max;

        IE_ASSERT(roundingType == "floor" || roundingType == "ceil" || roundingType == "");

        //
        // Update pads, strides, dilations
        //

        std::vector<int> padsBeginUpdate(kernelNDims);
        std::vector<int> padsEndUpdate(kernelNDims);
        std::vector<int> stridesUpdate(kernelNDims);

        for (int i = 0; i < kernelNDims; i++) {
            stridesUpdate[i] = strides.empty() ? 1 : strides[i];

            int remainder = inputShape[i + spacialDimsBegin] % stridesUpdate[i];
            int padsTotal = kernelShape[i] - (remainder? remainder: stridesUpdate[i]);

            if (autoPad == "") {
                padsBeginUpdate[i] = padsBegin[i];
                padsEndUpdate[i]   = padsEnd[i];
            } else if (autoPad == "valid") {
                padsBeginUpdate[i] = 0;
                padsEndUpdate[i]   = 0;
            } else if (autoPad == "same_lower") {
                padsEndUpdate[i]   = padsTotal / 2;
                padsBeginUpdate[i] = padsTotal - padsEndUpdate[i];
            } else if (autoPad == "same_upper") {
                padsBeginUpdate[i] = padsTotal / 2;
                padsEndUpdate[i]   = padsTotal - padsBeginUpdate[i];
            } else {
                IE_ASSERT(false); // this must never happen
            }
        }

        for (int i = 0; i < kernelNDims; i++) {
            int outputShapeExpected = (inputShape[i + spacialDimsBegin]
                                       + padsBeginUpdate[i] + padsEndUpdate[i]
                                       - kernelShape[i]
                                      ) / stridesUpdate[i] + 1;
            IE_ASSERT(outputShape[i + spacialDimsBegin] == outputShapeExpected);

        }

        int kernel_total = getTotalNum(kernelShape);
        int kernel_hits;

        //
        // Cycle over batch dimension (if any)
        //
        int N = batchNDims ? inputShape[0] : 1;
        for (int n = 0; n < N; n++) {
            std::vector<int> inputIndices(inputNDims);
            std::vector<int> outputIndices(outputNDims, 0);  // initialize with 0s
            if (batchNDims) {
                inputIndices[0] = n;
                outputIndices[0] = n;
            }

            //
            // Cycle over spacial dims of output tensor
            //
            do {
                //
                // Cycle over output channels
                //
                int C = outputChannels;
                for (int c = 0; c < C; c++) {
                    inputIndices[channelsDim] = c;
                    outputIndices[channelsDim] = c;

                    kernel_hits = 0;
                    float result = 0;
                    float value;

                    std::vector<int> kernelIndices(kernelNDims, 0);  // init with 0s

                    //
                    // Cycle over kernel
                    //
                    do {
                        //
                        // Setup spacial dims of inputIndices
                        //
                        for (int i = 0; i < kernelNDims; i++) {
                            int output_index_i = outputIndices[i + spacialDimsBegin];
                            int strided_output_index_i = output_index_i * stridesUpdate[i];

                            int index = strided_output_index_i
                                      + kernelIndices[i]
                                      - padsBeginUpdate[i];
                                
                            if (index < 0 || index >= inputShape[i + spacialDimsBegin]) {
                                goto nextKernelIndices;
                            }

                            inputIndices[i + spacialDimsBegin] = index;
                        }

                        value = input[offsetByIndex(inputIndices.data(), inputShape.data(), inputNDims)];

                        if (pooling_method == Avg) {
                            result = result + value;
                        } else {
                            result = std::max(result, value);
                        }

                        kernel_hits++;

                    nextKernelIndices:
                        continue;
                    } while (nextIndices(kernelIndices.data(), kernelShape.data(), kernelNDims));

                    if (pooling_method == Avg) {
                        if (excludePad) {
                            if (kernel_hits > 0) {
                                result = result / kernel_hits;
                            } else {
                                IE_ASSERT(result == 0);
                            }
                        } else {
                            result = result / kernel_total;
                        }
                    }

                    output[offsetByIndex(outputIndices.data(), outputShape.data(), outputNDims)] = result;
                }
            } while (nextIndices(&outputIndices[spacialDimsBegin],
                                 &outputShape[spacialDimsBegin],
                                  kernelNDims));
        }
    }

    static
    bool nextIndices(int indices[],
               const int shape[],
                     int nDims) {
        // let W's index change quicker than H's:
        // note that dims order is like ..., H, W
        for (int i = nDims - 1; i >= 0; i--) {
            if (++indices[i] < shape[i])
                return true;
            indices[i] = 0;
        }
        return false; // cannot get next indices
    }

    // Get element offset by ND index
    static
    int offsetByIndex(const int index[],
                      const int shape[],
                      const int ndims) {
        int offset = 0;
        int stride = 1;
        for (int i = ndims - 1; i >= 0; i--) {
            offset += index[i] * stride;
            stride *= shape[i];
        }
        return offset;
    }

    // Count total number of elements in ND tensor
    static
    int getTotalNum(const std::vector<int>& shape) {
        int totalNum = 1;
        for (int i = 0; i < shape.size(); i++) {
            totalNum *= shape[i];
        }
        return totalNum;
    }

    // Convert FP16 tensor data into FP32 format
    static
    void copyF16ToF32(const ie_fp16 f16Data[],
                            float   f32Data[],
                      const int     num) {
        for (int i = 0; i < num; i++) {
            f32Data[i] = PrecisionUtils::f16tof32(f16Data[i]);
        }
    }

    // Convert FP32 tensor data into FP16 format
    static
    void copyF32ToF16(const float   f32Data[],
                            ie_fp16 f16Data[],
                      const int     num) {
        for (int i = 0; i < num; i++) {
            f16Data[i] = PrecisionUtils::f32tof16(f32Data[i]);
        }
    }

    // Fulfill data[] array with random numbers
    // distributed uniformly in the interval [a,b]
    static
    void fulfillUniformly(uint8_t* data, int num, Precision precision, double a, double b) {
        IE_ASSERT(Precision::FP16 == precision);
        std::srand(1);
        for (int i = 0; i < num; i++) {
            double r = std::rand() / (double)RAND_MAX;
            float v = static_cast<float>(a*(1 - r) + b*r);
            reinterpret_cast<ie_fp16*>(data)[i] = PrecisionUtils::f32tof16(v);
        }
    }

    // Fulfill data[] array with random numbers,
    // Gaissian distribution with the given mean and standard deviation
    static
    void fulfillGaussian(uint8_t* data, int num, Precision precision,
                         double mean, double stdDev) {
        IE_ASSERT(Precision::FP16 == precision);
        std::srand(1);
        for (int i = 0; i < num; i++) {
            float value = static_cast<float>(randomGaussian(mean, stdDev));
            reinterpret_cast<ie_fp16*>(data)[i] = PrecisionUtils::f32tof16(value);
        }
    }

    // https://en.wikipedia.org/wiki/Marsaglia_polar_method
    static double randomGaussian(double mean, double stdDev) {
        static const double epsilon = std::numeric_limits<double>::min();
        thread_local static double spare, hasSpare = false;

        if (hasSpare) {
            hasSpare = false;
            return mean + stdDev * spare;
        }

        double u, v, s;
        do {
            u = rand() / static_cast<double>(RAND_MAX);
            v = rand() / static_cast<double>(RAND_MAX);
            s = u*u + v*v;
        } while (s > 1 || s < epsilon);
        s = std::sqrt(-2. * std::log(s) / s);

        spare = v * s;
        hasSpare = true;
        return mean + stdDev * (u * s);
    }

    static
    TBlob<uint8_t>::Ptr createPlainTBlob(const std::vector<int>& shape,
                                         const Precision& precision)
    {
        int ndims = shape.size();
        int length = 1;
        for (int i = 0; i < ndims; i++) {
            length *= shape[i];
        }
        SizeVector dims { length * precision.size() };
        Layout layout = Layout::ANY; // as plain memory
        TensorDesc tensorDesc(Precision::U8, dims, layout);
        TBlob<uint8_t>::Ptr blob = std::make_shared<TBlob<uint8_t>>(tensorDesc);
        return blob;
    }

    static
    std::string createModel(const std::vector<int>& inputShape,
                            const std::vector<int>& outputShape,
                            const std::vector<int>& kernelShape,
                            const std::vector<int>& strides,
                            const std::vector<int>& padsBegin,
                            const std::vector<int>& padsEnd,
                            const std::string       autoPad,
                            const std::string       poolingMethod,
                            const std::string       roundingType,
                            const bool              excludePad)
    {
        std::string model = R"V0G0N(
            <?xml version="1.0" ?>
            <net name="testPoolND" version="6">
                <layers>
                    <layer id="0" name="input" type="Input" precision="__PRECISION__">
                        <output>
                            <port id="0">
                                __INPUT_DIMS__
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="pooling" type="Pooling" precision="__PRECISION__">
                        <data kernel="__KERNEL__"
                              strides="__STRIDES__"
                              pads_begin="__PADS_BEGIN__"
                              pads_end="__PADS_END__"
                              auto_pad="__AUTO_PAD__"
                              pool-method="__POOLING_METHOD__"
                              rounding_type="__ROUNDING_TYPE__"
                              exclude-pad="__EXCLUDE_PAD__"
                        />
                        <input>
                            <port id="0">
                                __INPUT_DIMS__
                            </port>
                        </input>
                        <output>
                            <port id="1">
                                __OUTPUT_DIMS__
                            </port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                </edges>
            </net>
        )V0G0N";

        REPLACE_WITH_STR(model, "__PRECISION__", "FP16");

        const std::string inputDimsStr = shapeToDimsString(inputShape);
        const std::string outputDimsStr = shapeToDimsString(outputShape);
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", inputDimsStr);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", outputDimsStr);

        const std::string kernelShapeStr = shapeToString(kernelShape);
        REPLACE_WITH_STR(model, "__KERNEL__", kernelShapeStr);

        if (strides.empty()) {
            REPLACE_WITH_STR(model, "strides=\"__STRIDES__\"", "");
        } else {
            const std::string stridesStr = shapeToString(strides);
            REPLACE_WITH_STR(model, "__STRIDES__", stridesStr);
        }

        if (autoPad == "") {
            const std::string padsBeginStr = shapeToString(padsBegin);
            const std::string padsEndStr = shapeToString(padsEnd);
            REPLACE_WITH_STR(model, "__PADS_BEGIN__", padsBeginStr);
            REPLACE_WITH_STR(model, "__PADS_END__", padsEndStr);
            REPLACE_WITH_STR(model, "auto_pad=\"__AUTO_PAD__\"", "");
        } else {
            REPLACE_WITH_STR(model, "pads_begin=\"__PADS_BEGIN__\"", "");
            REPLACE_WITH_STR(model, "pads_end=\"__PADS_END__\"", "");
            REPLACE_WITH_STR(model, "__AUTO_PAD__", autoPad);
        }

        REPLACE_WITH_STR(model, "__POOLING_METHOD__", poolingMethod);

        if (roundingType == "") {
            REPLACE_WITH_STR(model, "rounding_type=\"__ROUNDING_TYPE__\"", "");
        } else {
            REPLACE_WITH_STR(model, "__ROUNDING_TYPE__", roundingType);
        }

        REPLACE_WITH_STR(model, "__EXCLUDE_PAD__", (excludePad? "true": "false"));

        return model;
    }

    static
    std::string shapeToString(const std::vector<int>& shape) {
        std::string str;
        for (int i = 0; i < shape.size(); i++) {
            str += (i? ", ": "");
            str += std::to_string(shape[i]);
        }
        return str;
    }

    static
    std::string shapeToDimsString(const std::vector<int>& shape)
    {
        std::string str;
        for (int i = 0; i < shape.size(); i++) {
            str += (i? " ": "");
            str += "<dim>" + std::to_string(shape[i]) + "</dim>";
        }
        return str;
    }
};

class myriadLayersPoolNDTest_smoke: public PoolNDTest {};

TEST_P(myriadLayersPoolNDTest_smoke, PoolND) {
    testPoolND();
}
