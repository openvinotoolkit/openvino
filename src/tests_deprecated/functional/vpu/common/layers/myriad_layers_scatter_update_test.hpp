// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//
//  Scatter Update layer is similar to Tensor Flow scatter_update operation
//  except we allow the additional `axis` parameter:
//
//  https://tensorflow.org/versions/r1.15/api_docs/python/tf/scatter_update
//
//  Yet, only axis == 0 is supported for our Scatter Update layer,
//  so that it literally implements this Tensor Flow operation
//
//  For example, the tensor shapes could be:
//  -    {N, C, H, W} for `input` and `output`
//  - {I, J, C, H, W} for `updates` tensor
//  - {I, J} for `indices`
//
//  Given some (i, j), the Scatter Update would copy the subtensor
//  `updates(i, j, :, :, :)` into the `output(n, :, :, :)`, where
//  `n = indices(i, j)`.
//

#include <myriad_layers_tests.hpp>
#include <vpu_case_common.hpp>

#include <algorithm>
#include <random>
#include <vector>

#include <cmath>
#include <cstring>

#define DEBUG 0

using namespace InferenceEngine;

using InputShape   = SizeVector;
using IndicesShape = SizeVector;

using ScatterUpdateTestParams = std::tuple<IndicesShape,
                                           InputShape>;

class myriadLayersScatterUpdateTest_smoke:
    public myriadLayerTestBaseWithParam<ScatterUpdateTestParams>
{
protected:

    void testScatterUpdate() {
        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        //
        // Get and verify test parameters, and deduce other parameters
        //

        const auto& params = GetParam();

        const SizeVector indicesShape = std::get<0>(params);
        const SizeVector   inputShape = std::get<1>(params);

        const int indicesNDims = indicesShape.size();
        const int   inputNDims =   inputShape.size();

        IE_ASSERT(inputNDims > 0);
        IE_ASSERT(indicesNDims > 0);

        // Exclude test if input tensor is too large for device with
        // less than 2 GB of RAM, i.e. for any one except 2085 board
        bool tooLarge = getTotal(inputShape) > 25 * 1000 * 1000;
        DISABLE_IF(tooLarge && !CheckMA2085());

        SizeVector outputShape = inputShape;  // copy
        const int outputNDims = inputNDims;

        SizeVector axisShape = {};

        // E.g.:
        //    {N, C, H, W} could be shape of `input` and `output`
        // {I, J, C, H, W} could be shape of `update` tensor
        // {I, J}          could be shape of `indices`
        SizeVector updatesShape = indicesShape;
        for (int i = 0; i < outputNDims - 1; i++) {
            updatesShape.push_back(outputShape[i + 1]);
        }

        //
        // Initialize input tensors, and compute reference output
        //

        const int inputTotal = getTotal(inputShape);
        const int outputTotal = getTotal(outputShape);
        const int indicesTotal = getTotal(indicesShape);
        const int updatesTotal = getTotal(updatesShape);
        const int axisTotal = getTotal(axisShape);

        std::vector<ie_fp16> inputData(inputTotal);
        std::vector<ie_fp16> outputData(outputTotal);
        std::vector<int32_t> indicesData(indicesTotal);
        std::vector<ie_fp16> updatesData(updatesTotal);
        std::vector<int32_t> axisData(axisTotal);

        std::mt19937 gen;

        fillUniformly(inputData.data(), inputTotal, Precision::FP16, 0, 255, gen);
        fillUniformly(updatesData.data(), updatesTotal, Precision::FP16, -1, +1, gen);

        const int indicesLimit = outputShape[0] - 1;
        fillUniformly(indicesData.data(), indicesTotal, Precision::I32, 0, indicesLimit, gen);

        axisData[0] = 0;  // yet we support only axis == 0

        referenceScatterUpdate(inputShape,
                               outputShape,
                               indicesShape,
                               updatesShape,
                               axisShape,
                               inputData,
                               outputData,
                               indicesData,
                               updatesData,
                               axisData);

        //
        // Initialize 1-layer network, and infer
        //

        std::string model = createModel(inputShape,
                                        outputShape,
                                        indicesShape,
                                        updatesShape);
        #if DEBUG
        std::cout << "model:\n" << model << "\n";
        #endif

        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(Precision::FP16);
        _inputsInfo["indices"]->setPrecision(Precision::I32);
        _inputsInfo["updates"]->setPrecision(Precision::FP16);
        _inputsInfo["axis"]->setPrecision(Precision::I32);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["scatter_update"]->setPrecision(Precision::FP16);

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, _config));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
        
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = _inferRequest.GetBlob("input"));
        
        void* inputBlobData = inputBlob->buffer();
        ASSERT_NE(inputBlobData, nullptr);
        std::copy(inputData.cbegin(), inputData.cend(), reinterpret_cast<ie_fp16*>(inputBlobData));

        Blob::Ptr indicesBlob;
        ASSERT_NO_THROW(indicesBlob = _inferRequest.GetBlob("indices"));
        
        void* indicesBlobData = indicesBlob->buffer();
        ASSERT_NE(indicesBlobData, nullptr);
        std::copy(indicesData.cbegin(), indicesData.cend(), reinterpret_cast<int32_t*>(indicesBlobData));

        Blob::Ptr updatesBlob;
        ASSERT_NO_THROW(updatesBlob = _inferRequest.GetBlob("updates"));
        
        void* updatesBlobData = updatesBlob->buffer();
        ASSERT_NE(updatesBlobData, nullptr);
        std::copy(updatesData.cbegin(), updatesData.cend(), reinterpret_cast<ie_fp16*>(updatesBlobData));

        Blob::Ptr axisBlob;
        ASSERT_NO_THROW(axisBlob = _inferRequest.GetBlob("axis"));
        
        void* axisBlobData = axisBlob->buffer();
        ASSERT_NE(axisBlobData, nullptr);
        std::copy(axisData.cbegin(), axisData.cend(), reinterpret_cast<int32_t*>(axisBlobData));

        ASSERT_NO_THROW(_inferRequest.Infer());
        
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("scatter_update"));
        
        const void* outputBlobDataPtr = outputBlob->cbuffer();
        const ie_fp16* outputBlobData = reinterpret_cast<const ie_fp16*>(outputBlobDataPtr);
        ASSERT_NE(outputBlobData, nullptr);

        //
        // Check result
        //

        int errors = 0;

        // cycle over `output` coordinates
        SizeVector outputCoord(outputNDims, 0);
        do {
            const int outputOffset = offsetByCoord(outputCoord.data(), outputShape.data(), outputNDims);

            const float result = PrecisionUtils::f16tof32(outputBlobData[outputOffset]);
            const float reference = PrecisionUtils::f16tof32(outputData[outputOffset]);
            const float diff = result - reference;

            if (diff != 0) {
                if (errors++ < 25) {
                    std::cout << "error:"
                        << " outputCoord=" << to_string(outputCoord)
                        << " result=" << result
                        << " reference=" << reference
                        << " diff=" << diff
                        << std::endl;
                }
            }
        } while (nextCoord(outputCoord.data(), outputShape.data(), outputNDims));

        if (errors > 0) {
            std::cout << "errors: " << errors << std::endl;
        }

        ASSERT_EQ(errors, 0);
    }

private:

    static
    void referenceScatterUpdate(const      SizeVector     & inputShape,
                                const      SizeVector     & outputShape,
                                const      SizeVector     & indicesShape,
                                const      SizeVector     & updatesShape,
                                const      SizeVector     & axisShape,
                                const std::vector<ie_fp16>& inputData,
                                      std::vector<ie_fp16>& outputData,
                                const std::vector<int32_t>& indicesData,
                                const std::vector<ie_fp16>& updatesData,
                                const std::vector<int32_t>& axisData) {
        // yet we only support axis == 0
        IE_ASSERT(axisShape.size() == 0 ||
                  axisShape.size() == 1);
        if (axisShape.size() > 0) {
            IE_ASSERT(axisShape[0] == 1);
        }
        IE_ASSERT(axisData[0] == 0);

        // copy `input` to `output`
        const int inputTotal = getTotal(inputShape);
        const int outputTotal = getTotal(outputShape);
        IE_ASSERT(inputTotal == outputTotal);
        std::copy(inputData.cbegin(), inputData.cend(), outputData.begin());

        const int outputNDims = outputShape.size();
        SizeVector outputCoord(outputNDims, 0);

        // cycle over indices of `updates` tensor
        const int updatesNDims = updatesShape.size();
        SizeVector updatesCoord(updatesNDims, 0);
        do {
            const int indicesNDims = indicesShape.size();
            const size_t* indicesCoord = updatesCoord.data();
            const int indicesOffset = offsetByCoord(indicesCoord, indicesShape.data(), indicesNDims);
            const int n = indicesData[indicesOffset];

            const int axis = 0;
            IE_ASSERT(0 <= n && n < outputShape[axis]);

            for (int i = 0; i < outputNDims - 1; i++) {
                outputCoord[i + 1] = updatesCoord[i + indicesNDims];
            }
            outputCoord[0] = n;

            const int outputOffset = offsetByCoord(outputCoord.data(), outputShape.data(), outputNDims);
            const int updatesOffset = offsetByCoord(updatesCoord.data(), updatesShape.data(), updatesNDims);

            const ie_fp16 value = updatesData[updatesOffset];
            outputData[outputOffset] = value;
        } while (nextCoord(updatesCoord.data(), updatesShape.data(), updatesNDims));
    }

    static
    std::string to_string(const SizeVector& v) {
        std::stringstream s;
        s << "{";
        for (int i = 0; i < v.size(); i++) {
            s << (i? ", ": "") << v[i];
        }
        s << "}";
        return s.str();
    }

    static
    bool nextCoord(size_t coord[],
             const size_t shape[],
                   int    nDims) {
        // let W's index change quicker than H's:
        // note that dims order is like ..., H, W
        for (int i = nDims - 1; i >= 0; i--) {
            if (++coord[i] < shape[i])
                return true;
            coord[i] = 0;
        }
        return false; // cannot get next indices
    }

    // Get element offset by ND coordinates
    static
    int offsetByCoord(const size_t coord[],
                      const size_t shape[],
                      const int    ndims) {
        int offset = 0;
        int stride = 1;
        for (int i = ndims - 1; i >= 0; i--) {
            offset += coord[i] * stride;
            stride *= shape[i];
        }
        return offset;
    }

    // Count total number of elements in ND tensor
    static
    int getTotal(const SizeVector& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    // Fill data[] array with random numbers
    // distributed uniformly in the interval [a,b]
    static
    void fillUniformly(void* data,
                       const int num,
                       const Precision& precision,
                       const double a,
                       const double b,
                       std::mt19937& gen) {
        if (Precision::FP16 == precision) {
            std::uniform_real_distribution<float> uniform(a, b);
            for (int i = 0; i < num; i++) {
                const float v = uniform(gen);
                reinterpret_cast<ie_fp16*>(data)[i] = PrecisionUtils::f32tof16(v);
            }
        } else
        if (Precision::I32 == precision) {
            const int ia = static_cast<int>(std::round(a));
            const int ib = static_cast<int>(std::round(b));
            std::uniform_int_distribution<int> uniform(ia, ib);
            for (int i = 0; i < num; i++) {
                const int v = uniform(gen);
                reinterpret_cast<int32_t*>(data)[i] = v;
            }
        } else {
            IE_ASSERT(Precision::FP16 == precision ||
                        Precision::I32  == precision);
        }
    }

    // Note that:
    // - IR version is v7 (should be v10): as readNetwork() method
    //   cannot parse / denies IR v10 if there's no weights tensor
    static
    std::string createModel(const SizeVector& inputShape,
                            const SizeVector& outputShape,
                            const SizeVector& indicesShape,
                            const SizeVector& updatesShape) {
        std::string model = R"V0G0N(
            <?xml version="1.0" ?>
            <net name="testScatterUpdate" version="7">
                <layers>
                    <layer id="0" name="input" type="Input">
                        <output>
                            <port id="0" precision="FP16">
                                __INPUT_DIMS__
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="indices" type="Input">
                        <output>
                            <port id="0" precision="I32">
                                __INDICES_DIMS__
                            </port>
                        </output>
                    </layer>
                    <layer id="2" name="updates" type="Input">
                        <output>
                            <port id="0" precision="FP16">
                                __UPDATES_DIMS__
                            </port>
                        </output>
                    </layer>
                    <layer id="3" name="axis" type="Input">
                        <output>
                            <port id="0" precision="I32">
                            </port>
                        </output>
                    </layer>
                    <layer id="4" name="scatter_update" type="ScatterUpdate">
                        <input>
                            <port id="0" precision="FP16">
                                __INPUT_DIMS__
                            </port>
                            <port id="1" precision="I32">
                                __INDICES_DIMS__
                            </port>
                            <port id="2" precision="FP16">
                                __UPDATES_DIMS__
                            </port>
                            <port id="3" precision="I32">
                            </port>
                        </input>
                        <output>
                            <port id="4" precision="FP16">
                                __OUTPUT_DIMS__
                            </port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
                    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
                    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
                    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
                </edges>
            </net>
        )V0G0N";

        const std::string inputDimsStr = shapeToDimsString(inputShape);
        const std::string outputDimsStr = shapeToDimsString(outputShape);
        const std::string indicesDimsStr = shapeToDimsString(indicesShape);
        const std::string updatesDimsStr = shapeToDimsString(updatesShape);
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", inputDimsStr);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", outputDimsStr);
        REPLACE_WITH_STR(model, "__INDICES_DIMS__", indicesDimsStr);
        REPLACE_WITH_STR(model, "__UPDATES_DIMS__", updatesDimsStr);

        return model;
    }

    static
    std::string shapeToDimsString(const SizeVector& shape)
    {
        std::string str;
        for (int i = 0; i < shape.size(); i++) {
            str += (i? " ": "");
            str += "<dim>" + std::to_string(shape[i]) + "</dim>";
        }
        return str;
    }
};

TEST_P(myriadLayersScatterUpdateTest_smoke, accuracy) {
    testScatterUpdate();
}
