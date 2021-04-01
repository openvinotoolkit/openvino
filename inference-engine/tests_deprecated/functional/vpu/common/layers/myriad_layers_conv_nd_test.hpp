// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_layers_tests.hpp"
#include "vpu_case_common.hpp"
#include "precision_utils.h"

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define DEBUG 0

using namespace InferenceEngine;

using InputShape  = std::vector<int>;
using KernelShape = std::vector<int>;
using PadsBegin   = std::vector<int>;
using PadsEnd     = std::vector<int>;
using AutoPad     = std::string;
using Strides     = std::vector<int>;
using Dilations   = std::vector<int>;
using OutputChannels = int;
using Groups         = int;

using ConvNDTestParams =
    std::tuple<
        InputShape,
        KernelShape,
        PadsBegin,
        PadsEnd,
        AutoPad,
        Strides,
        Dilations,
        OutputChannels,
        Groups>;

class ConvNDTest: public myriadLayerTestBaseWithParam<ConvNDTestParams>
{
protected:

    void testConvND() {
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        _config[InferenceEngine::MYRIAD_HW_INJECT_STAGES]     = CONFIG_VALUE(NO);

        //
        // Get test parameters
        //

        const auto& params = GetParam();

        const std::vector<int>  inputShape = std::get<0>(params);
        const std::vector<int> kernelShape = std::get<1>(params);
        const std::vector<int>   padsBegin = std::get<2>(params);
        const std::vector<int>   padsEnd   = std::get<3>(params);
        const std::string          autoPad = std::get<4>(params);
        const std::vector<int>     strides = std::get<5>(params);
        const std::vector<int>   dilations = std::get<6>(params);
        const int           outputChannels = std::get<7>(params);
        const int                   groups = std::get<8>(params);

        // Exclude the `i3d_id6` test case, which requires at least
        // 1 GB of RAM on device, like e.g. ma2085 board
        bool tooLarge = kernelShape[0] == 7 &&
                        inputShape[inputShape.size() - 1] == 224;
        DISABLE_IF(tooLarge && !CheckMA2085());

        //
        // TODO: Add `withBiases` to test parameters
        //
        const bool withBiases  = true;

        // Only support not-interleaved layouts: CHW, NCHW, NCDHW, ...
        const bool interleaved = false;

        const int     inputNDims =  inputShape.size();
        const int    kernelNDims = kernelShape.size();
        const int   stridesNDims =     strides.size();
        const int dilationsNDims =   dilations.size();
        const int padsBeginNDims =   padsBegin.size();
        const int padsEndNDims   =   padsEnd.size();

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

        IE_ASSERT(groups > 0);
        IE_ASSERT(inputChannels > 0);
        IE_ASSERT(outputChannels > 0);
        IE_ASSERT(inputChannels % groups == 0);
        IE_ASSERT(outputChannels % groups == 0);

        IE_ASSERT(kernelNDims > 0);
        IE_ASSERT(kernelNDims == stridesNDims   || stridesNDims == 0);
        IE_ASSERT(kernelNDims == dilationsNDims || dilationsNDims == 0);

        IE_ASSERT(autoPad == "same_lower" ||
                  autoPad == "same_upper" ||
                  autoPad == "valid" ||
                  autoPad == "");

        if (autoPad == "") {
            IE_ASSERT(kernelNDims == padsBeginNDims);
            IE_ASSERT(kernelNDims == padsEndNDims);
        } else {
            IE_ASSERT(0 == padsBeginNDims);
            IE_ASSERT(0 == padsEndNDims);
        }

        //
        // Derive other parameters of layer
        //

        std::vector<int> padsBeginUpdate(kernelNDims);
        std::vector<int> padsEndUpdate(kernelNDims);
        std::vector<int> stridesUpdate(kernelNDims);
        std::vector<int> dilationsUpdate(kernelNDims);
        std::vector<int> dilatedKernelShape(kernelNDims);

        std::vector<int> outputShape(inputNDims);
        for (int i = 0; i < kernelNDims; i++) {
            stridesUpdate[i] = stridesNDims ? strides[i] : 1;
            dilationsUpdate[i] = dilationsNDims ? dilations[i] : 1;
            dilatedKernelShape[i] = dilationsUpdate[i] * (kernelShape[i] - 1) + 1;

            int remainder_i = inputShape[i + spacialDimsBegin] % stridesUpdate[i];
            int pads_i = dilatedKernelShape[i] - (remainder_i? remainder_i: stridesUpdate[i]);

            if (autoPad == "") {
                padsBeginUpdate[i] = padsBegin[i];
                padsEndUpdate[i]   = padsEnd[i];
            } else if (autoPad == "valid") {
                padsBeginUpdate[i] = 0;
                padsEndUpdate[i]   = 0;
            } else if (autoPad == "same_lower") {
                padsEndUpdate[i]   = pads_i / 2;                 // floor(pads_i / 2.)
                padsBeginUpdate[i] = pads_i - padsEndUpdate[i];  //  ceil(pads_i / 2.)
            } else if (autoPad == "same_upper") {
                padsBeginUpdate[i] = pads_i / 2;
                padsEndUpdate[i]   = pads_i - padsBeginUpdate[i];
            } else {
                IE_ASSERT(false); // this must never happen
            }

            outputShape[i + spacialDimsBegin] =
                (inputShape[i + spacialDimsBegin]
                + padsBeginUpdate[i] + padsEndUpdate[i]
                - dilatedKernelShape[i]
                ) / stridesUpdate[i] + 1;
        }
        outputShape[channelsDim] = outputChannels;
        if (batchNDims) {
            outputShape[0] = inputShape[0]; // copy batch size
        }

        std::vector<int> weightsShape(kernelNDims + 2);
        for (int i = 0; i < kernelNDims; i++) {
            weightsShape[i + 2] = kernelShape[i];
        }
        weightsShape[1] =  inputChannels / groups;
        weightsShape[0] = outputChannels / groups;

        std::vector<int> biasesShape {outputChannels / groups};

        //
        // Initialize data
        //

        TBlob<uint8_t>::Ptr inputBlob = createPlainTBlob(inputShape, Precision::FP16);
        TBlob<uint8_t>::Ptr outputBlob = createPlainTBlob(outputShape, Precision::FP16);

        int weightsNum = 1;
        for (int i = 0; i < weightsShape.size(); i++) {
            weightsNum *= weightsShape[i];
        }
        int biasesNum = outputChannels / groups;
        int coeffsNum = weightsNum + biasesNum;
        std::vector<int> coeffsShape { coeffsNum };

        TBlob<uint8_t>::Ptr coeffsBlob = createPlainTBlob(coeffsShape, Precision::FP16);

        inputBlob->allocate();
        outputBlob->allocate();
        coeffsBlob->allocate();

        uint8_t* inputBlobDataPtr = inputBlob->data();
        uint8_t* coeffsBlobDataPtr = coeffsBlob->data();

        int inputNum = getTotalNum(inputShape);

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
        fulfillUniformly(coeffsBlobDataPtr, coeffsNum, Precision::FP16, -1, 1);
        #else
        // Coefficients to simulate average pooling, although with random deviations
        double coeffsAvg = 1. / getTotalNum(kernelShape) / (inputChannels / groups);
        double coeffsDev = coeffsAvg * 0.5;  // 50% deviation
        fulfillGaussian(inputBlobDataPtr, inputNum, Precision::FP16, 128, 32);
        fulfillGaussian(coeffsBlobDataPtr, coeffsNum, Precision::FP16, coeffsAvg, coeffsDev);
        #endif

        //
        // Initialize network
        //

        std::string model = createModel(inputShape,
                                        kernelShape,
                                        padsBegin,
                                        padsEnd,
                                        autoPad,
                                        strides,
                                        dilations,
                                        groups,
                                        outputShape,
                                        weightsShape,
                                        biasesShape);
        #if DEBUG
        std::cout << "model:\n" << model << "\n";
        #endif

        ASSERT_NO_THROW(readNetwork(model, coeffsBlob));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["convolution"]->setPrecision(Precision::FP16);

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
        ASSERT_NO_THROW(outputValuesBlob = _inferRequest.GetBlob("convolution"));
        
        //
        // Check result
        //

        const ie_fp16 *inputData = inputValuesBlob->cbuffer().as<ie_fp16*>();
        const ie_fp16 *outputData = outputValuesBlob->cbuffer().as<ie_fp16*>();

        const ie_fp16* weightsData = (const ie_fp16*) coeffsBlobDataPtr;
        const ie_fp16* biasesData = weightsData + weightsNum;

        int  outputNDims = inputNDims;
        int weightsNDims = kernelNDims + 2;

        std::vector<int> inputIndices(inputNDims);
        std::vector<int> outputIndices(outputNDims);
        std::vector<int> weightsIndices(weightsNDims);

        // Check ~10000 of output points (part maybe out of millions):
        // given a point, check it with probability 10000/totalOutputs
        int totalOutputs = getTotalNum(outputShape);
        double P = std::min(1., 100000. / totalOutputs);
        std::uniform_real_distribution<> uniform(0, 1);
        std::mt19937 gen;

        int points = 0;  // count: points actually checked
        int errors = 0;  // count: errors exceeded tolerance

        float tolerance = 0.01;  // 1%
        float avgRelDif = 0;     // count: average of relative diff

        //
        // Cycle over batch dimension (if any)
        //
        int N = batchNDims ? inputShape[0] : 1;
        for (int n = 0; n < N; n++) {
            if (batchNDims > 0) {
                inputIndices[0] = n;
                outputIndices[0] = n;
            }

            //
            // Cycle over spacial dims of output
            //
            do {
                //
                // Cycle over output channels
                //
                int C = outputChannels;
                for (int c = 0; c < C; c++) {
                    outputIndices[channelsDim] = c;

                    // check with probability P
                    double p = uniform(gen);
                    if (p > P) {
                        continue;
                    }
                    points++;

                    float reference = referenceConvND(inputIndices,
                                                      outputIndices,
                                                      weightsIndices,
                                                      inputData,
                                                      weightsData,
                                                      biasesData,
                                                      inputShape,
                                                      outputShape,
                                                      weightsShape,
                                                      padsBeginUpdate,
                                                      stridesUpdate,
                                                      dilationsUpdate,
                                                      groups,
                                                      interleaved,
                                                      withBiases);

                    int resOffset = offsetByIndex(&outputIndices[0], &outputShape[0], outputNDims);
                    float result = PrecisionUtils::f16tof32(outputData[resOffset]);

                    float diff = result - reference;
                    float relative = std::fabs(diff) / std::fabs(reference);
                    if (relative > tolerance) {
                        if (errors++ < 25) {
                            std::cout << "error:"
                                << " outputIndices=" << to_string(outputIndices)
                                << " result=" << result
                                << " reference=" << reference
                                << " diff=" << diff
                                << " relative=" << to_percents(relative)
                                << std::endl;
                        }
                    }

                    avgRelDif += relative;  // accumulating...
                }
            } while (nextIndices(&outputIndices[spacialDimsBegin],
                                 &outputShape[spacialDimsBegin],
                                 kernelNDims));
        }

        if (points == 0) {
            FAIL() << "test bug: number of tested points must be (much!) greater than zero";
        }
        avgRelDif = avgRelDif / points;

        if (errors > 0) {
            std::cout << "errors: " << errors << " (tested points: " << points << ")" << std::endl;
            std::cout << "avgDif: " << to_percents(avgRelDif) << " (tolerance: " << to_percents(tolerance) << ")"
                      << std::endl;
        }

        ASSERT_LE(avgRelDif, tolerance);
    }

private:

    static
    std::string to_percents(float x) {
        std::stringstream s;
        s << std::setprecision(3);
        s << x * 100;
        s << "%";
        return s.str();
    }

    static
    std::string to_string(const std::vector<int>& v) {
        std::stringstream s;
        s << "{";
        for (int i = 0; i < v.size(); i++) {
            s << (i? ", ": "") << v[i];
        }
        s << "}";
        return s.str();
    }

    // Return result of ND convolution for the given output indices
    static
    float referenceConvND(std::vector<int>   & inputIndices,
                    const std::vector<int>   & outputIndices,
                          std::vector<int>   & weightsIndices,
                    const             ie_fp16  inputData[],
                    const             ie_fp16  weightsData[],
                    const             ie_fp16  biasesData[],
                    const std::vector<int>   & inputShape,
                    const std::vector<int>   & outputShape,
                    const std::vector<int>   & weightsShape,
                    const std::vector<int>   & padsBeginUpdate,
                    const std::vector<int>   & stridesUpdate,
                    const std::vector<int>   & dilationsUpdate,
                    const int                  groups,
                    const bool                 interleaved,
                    const bool                 withBiases)
    {
        const int  inputNDims =  inputShape.size();
        const int outputNDims = outputShape.size();
        const int weightsNDims = weightsShape.size();

        const int kernelNDims = weightsNDims - 2;

        IE_ASSERT(inputNDims == outputNDims);
        IE_ASSERT(inputNDims >= 3); // CHW, NCHW, NCDHW, ...

        const int channelsNDims = 1;
        const int batchNDims = inputNDims > 3; // 0 if CHW, 1 if NCHW etc
        IE_ASSERT(inputNDims == kernelNDims + channelsNDims + batchNDims);

        int padsBeginNDims = padsBeginUpdate.size();
        int   stridesNDims =   stridesUpdate.size();
        int dilationsNDims = dilationsUpdate.size();

        IE_ASSERT(kernelNDims > 0);
        IE_ASSERT(kernelNDims == padsBeginNDims);
        IE_ASSERT(kernelNDims == stridesNDims);
        IE_ASSERT(kernelNDims == dilationsNDims);

        const int channelsDim      = interleaved ? inputNDims - 1 : batchNDims;
        const int spacialDimsBegin = interleaved ? batchNDims     : channelsDim + 1;

        const int  inputChannels =  inputShape[channelsDim];
        const int outputChannels = outputShape[channelsDim];

        IE_ASSERT(weightsShape[0] == outputChannels / groups);
        IE_ASSERT(weightsShape[1] ==  inputChannels / groups);

        IE_ASSERT(groups > 0);
        IE_ASSERT(inputChannels > 0);
        IE_ASSERT(outputChannels > 0);
        IE_ASSERT(inputChannels % groups == 0);
        IE_ASSERT(outputChannels % groups == 0);

        int IC =  inputChannels / groups;
        int OC = outputChannels / groups;

        int c = outputIndices[channelsDim];

        int g  = c / OC;  // group of channels
        int oc = c % OC;  // channel of group

        // accumulate result with FP32 precision
        float result = withBiases ? PrecisionUtils::f16tof32(biasesData[oc]) : 0;

        for (int i = 0; i < kernelNDims; i++) {
            weightsIndices[i + 2] = 0;
        }
        weightsIndices[0] = oc;
    //  weightsIndices[1] = ic; -- defer till inner cycle by ic (below)

        //
        // Cycle over weights spacial indices, i.e. 2nd, 3rd, ...
        //
        do {
            //
            // Setup spacial dims of inputIndices
            //
            bool offside = false;
            for (int i = 0; i < kernelNDims; i++) {
                int index = outputIndices[i + spacialDimsBegin] * stridesUpdate[i]
                            + weightsIndices[i + 2] * dilationsUpdate[i]
                            - padsBeginUpdate[i];

                if (index < 0 || index >= inputShape[i + spacialDimsBegin]) {
                    offside = true;  // out of input tensor bounds,
                    break;           // so skip this weightsIndices
                }

                inputIndices[i + spacialDimsBegin] = index;
            }
            if (offside) {
                continue;  // goto next weightsIndices
            }

            //
            // Cycle over input channels in the group
            //
            for (int ic = 0; ic < IC; ic++) {
                inputIndices[channelsDim] = ic + g*IC;
                weightsIndices[1] = ic;
                ie_fp16 in = inputData[offsetByIndex(&inputIndices[0], &inputShape[0], inputNDims)];
                ie_fp16 w = weightsData[offsetByIndex(&weightsIndices[0], &weightsShape[0], weightsNDims)];
                result += PrecisionUtils::f16tof32(in) * PrecisionUtils::f16tof32(w);
            }
        } while (nextIndices(&weightsIndices[2],
                             &weightsShape[2],
                             kernelNDims));

        return result;
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

    // Fulfill data[] array with random numbers
    // distributed uniformly in the interval [a,b]
    static
    void fulfillUniformly(uint8_t* data, int num, Precision precision,
                          double a, double b) {
        IE_ASSERT(Precision::FP16 == precision);
        std::mt19937 gen;
        std::uniform_real_distribution<float> uniform(a, b);
        for (int i = 0; i < num; i++) {
            float v = uniform(gen);
            reinterpret_cast<ie_fp16*>(data)[i] = PrecisionUtils::f32tof16(v);
        }
    }

    // Fulfill data[] array with random numbers,
    // Gaissian distribution with the given mean and standard deviation
    static
    void fulfillGaussian(uint8_t* data, int num, Precision precision,
                         double mean, double stdDev) {
        IE_ASSERT(Precision::FP16 == precision);
        std::mt19937 gen;
        std::normal_distribution<float> gauss(mean, stdDev);
        for (int i = 0; i < num; i++) {
            float value = gauss(gen);
            reinterpret_cast<ie_fp16*>(data)[i] = PrecisionUtils::f32tof16(value);
        }
    }

    static
    TBlob<uint8_t>::Ptr createPlainTBlob(const std::vector<int>& shape,
                                         const Precision& precision)
    {
        int length = getTotalNum(shape);
        SizeVector dims { length * precision.size() };
        Layout layout = Layout::ANY; // as plain memory
        TensorDesc tensorDesc(Precision::U8, dims, layout);
        TBlob<uint8_t>::Ptr blob = std::make_shared<TBlob<uint8_t>>(tensorDesc);
        return blob;
    }

    static
    std::string createModel(const std::vector<int>& inputShape,
                            const std::vector<int>& kernelShape,
                            const std::vector<int>& padsBegin,
                            const std::vector<int>& padsEnd,
                            const std::string       autoPad,
                            const std::vector<int>& strides,
                            const std::vector<int>& dilations,
                            const int               groups,
                            const std::vector<int>& outputShape,
                            const std::vector<int>& weightsShape,
                            const std::vector<int>& biasesShape)
    {
        std::string model = R"V0G0N(
            <?xml version="1.0" ?>
            <net name="testConvND" version="6">
                <layers>
                    <layer id="0" name="input" type="Input" precision="__PRECISION__">
                        <output>
                            <port id="0">
                                __INPUT_DIMS__
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="convolution" type="Convolution" precision="__PRECISION__">
                        <data auto_pad="__AUTO_PAD__"
                              dilations="__DILATIONS__"
                              group="__GROUP__"
                              kernel="__KERNEL__"
                              output="__OUTPUT_CHANNELS__"
                              pads_begin="__PADS_BEGIN__"
                              pads_end="__PADS_END__"
                              strides="__STRIDES__"
                        />
                        <input>
                            <port id="0">
                                __INPUT_DIMS__
                            </port>
                        </input>
                        <output>
                            <port id="3">
                                __OUTPUT_DIMS__
                            </port>
                        </output>
                        <blobs>
                            <weights offset="0" size="__WEIGHTS_BYTES__"/>
                            <biases offset="__WEIGHTS_BYTES__" size="__BIASES_BYTES__"/>
                        </blobs>
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

        const std::string groupStr = std::to_string(groups);
        const std::string kernelShapeStr = shapeToString(kernelShape);
        REPLACE_WITH_STR(model, "__GROUP__", groupStr);
        REPLACE_WITH_STR(model, "__KERNEL__", kernelShapeStr);

        const int batchNDims = inputShape.size() > 3; // NCHW, NCDHW, ...
        const int channelsDim = batchNDims;
        const int outputChannels = outputShape[channelsDim];
        const std::string outputChannelsStr = std::to_string(outputChannels);
        REPLACE_WITH_STR(model, "__OUTPUT_CHANNELS__", outputChannelsStr);

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

        if (dilations.empty()) {
            REPLACE_WITH_STR(model, "dilations=\"__DILATIONS__\"", "");
        } else {
            const std::string dilationsStr = shapeToString(dilations);
            REPLACE_WITH_STR(model, "__DILATIONS__", dilationsStr);
        }

        if (strides.empty()) {
            REPLACE_WITH_STR(model, "strides=\"__STRIDES__\"", "");
        } else {
            const std::string stridesStr = shapeToString(strides);
            REPLACE_WITH_STR(model, "__STRIDES__", stridesStr);
        }

        int weightsElements = 1;
        for (int i = 0; i < weightsShape.size(); i++) {
            weightsElements *= weightsShape[i];
        }
        const int weightsBytes = weightsElements * sizeof(ie_fp16);
        const std::string weightsBytesStr = std::to_string(weightsBytes);
        REPLACE_WITH_STR(model, "__WEIGHTS_BYTES__", weightsBytesStr);

        const int biasesBytes = (outputChannels / groups) * sizeof(ie_fp16);
        const std::string biasesBytesStr = std::to_string(biasesBytes);
        REPLACE_WITH_STR(model, "__BIASES_BYTES__", biasesBytesStr);

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

class myriadLayersConvNDTest_smoke: public ConvNDTest {};

TEST_P(myriadLayersConvNDTest_smoke, ConvND) {
    testConvND();
}
