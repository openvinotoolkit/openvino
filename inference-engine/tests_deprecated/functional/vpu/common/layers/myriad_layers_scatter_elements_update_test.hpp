// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#include "vpu_case_common.hpp"
#include "precision_utils.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <string>

//----------------------------------------------------------

static
std::ostream& operator << (std:: ostream& out,
                     const std::vector<int>& shape) {
    out << "{";
    const int ndims = shape.size();
    for (int i = 0; i < ndims; i++) {
        if (i > 0) {
            out << ", ";
        }
        out << shape[i];
    }
    out << "}";
    return out;
}

//----------------------------------------------------------

using namespace InferenceEngine;

using DataShape = std::vector<int>;
using DataType  = std::string;  // "FP16", "I32"

using ScatterElementsUpdateTestParams = std::tuple<DataShape,
                                                   DataType>;

class myriadLayersScatterElementsUpdateTest_smoke :
    public myriadLayerTestBaseWithParam<ScatterElementsUpdateTestParams> {
protected:

    void testScatterElementsUpdate() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        //
        // Parse test parameters
        //

        const ScatterElementsUpdateTestParams& params = GetParam();
        const std::vector<int>& dataShape = std::get<0>(params);
        const std::string     & dataType  = std::get<1>(params);

        IE_ASSERT(dataType == "I32" ||
                  dataType == "FP16");

        const int dataNDims = dataShape.size();
        IE_ASSERT(dataNDims > 0);

        //
        // Random axis
        //

        std::uniform_int_distribution<int> axisDistr(0, dataNDims - 1);
        const int axis = axisDistr(m_gen);

        //
        // Random shape for indices and updates
        //

        std::vector<int> updatesShape(dataNDims);
        for (int i = 0; i < dataNDims; i++) {
            std::uniform_int_distribution<int> distr(1, dataShape[i]);
            updatesShape[i] = distr(m_gen);
        }

        //
        // Skip if data is too large
        //

        const int dataTotal = getTotal(dataShape);
        const int updatesTotal = getTotal(updatesShape);

        const int bpp = dataType == "I32" ? sizeof(int32_t) : sizeof(ie_fp16);
        const int dataByteLength = dataTotal * bpp;

        const int dataByteLengthThreshold = 30 * (1 << 20);  // 30 MB

        const bool tooLarge = dataByteLength > dataByteLengthThreshold;

        // Disabling large-data tests at all even for PrismCreek (ma2085). See:
        // #-30792 [VPU] re-enable Scatter Elements Update tests for Prism Creek
        DISABLE_IF(tooLarge);  // TODO: fix tests and re-enable if CheckMA2085()

        //
        // Initialize 1-layer network
        //

        std::string model = createModel(dataType, dataShape, updatesShape);

        ASSERT_NO_THROW(readNetwork(model));

        Precision precision = dataType == "I32" ? Precision::I32 : Precision::FP16;

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(precision);
        _inputsInfo["updates"]->setPrecision(precision);
        _inputsInfo["indices"]->setPrecision(Precision::I32);
        _inputsInfo["axis"]->setPrecision(Precision::I32);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["scatter"]->setPrecision(precision);

        //
        // Create infer request and get its blobs pointers
        //

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network, _config));
        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
        
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = _inferRequest.GetBlob("input"));
        
        Blob::Ptr indicesBlob;
        ASSERT_NO_THROW(indicesBlob = _inferRequest.GetBlob("indices"));
        
        Blob::Ptr updatesBlob;
        ASSERT_NO_THROW(updatesBlob = _inferRequest.GetBlob("updates"));
        
        Blob::Ptr axisBlob;
        ASSERT_NO_THROW(axisBlob = _inferRequest.GetBlob("axis"));
        
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = _inferRequest.GetBlob("scatter"));
        
        Blob::Ptr referenceBlob;
        if (dataType == "I32") {
            referenceBlob = make_shared_blob<int32_t>(outputBlob->getTensorDesc());
        } else {
            referenceBlob = make_shared_blob<ie_fp16>(outputBlob->getTensorDesc());
        }
        referenceBlob->allocate();

        //
        // Initialize blobs: `input`, `indices`, `updates` and `axis`
        //

        void* inputBlobData = inputBlob->buffer();
        ASSERT_NE(inputBlobData, nullptr);

        void* indicesBlobData = indicesBlob->buffer();
        ASSERT_NE(indicesBlobData, nullptr);

        void* updatesBlobData = updatesBlob->buffer();
        ASSERT_NE(indicesBlobData, nullptr);

        void* axisBlobData = axisBlob->buffer();
        ASSERT_NE(axisBlobData, nullptr);

        const int indicesLimit = dataShape[axis] - 1;

        fillUniformly(inputBlobData, dataTotal, precision, 0, 50000, m_gen);
        fillUniformly(updatesBlobData, updatesTotal, precision, 0, 50000, m_gen);
        fillUniformly(indicesBlobData, updatesTotal, Precision::I32, 0, indicesLimit, m_gen);

        reinterpret_cast<int32_t*>(axisBlobData)[0] = axis;

        //
        // Infer
        //

        const auto layoutPreference = vpu::LayoutPreference::ChannelMajor;

        const auto inputLayout = inputBlob->getTensorDesc().getLayout();
        const auto outputLayout = outputBlob->getTensorDesc().getLayout();
        const auto indicesLayout = indicesBlob->getTensorDesc().getLayout();
        const auto updatesLayout = updatesBlob->getTensorDesc().getLayout();

        inputBlob->getTensorDesc().setLayout(vpu::deviceLayout(inputLayout, layoutPreference));
        indicesBlob->getTensorDesc().setLayout(vpu::deviceLayout(indicesLayout, layoutPreference));
        updatesBlob->getTensorDesc().setLayout(vpu::deviceLayout(updatesLayout, layoutPreference));
        outputBlob->getTensorDesc().setLayout(vpu::deviceLayout(outputLayout, layoutPreference));
        referenceBlob->getTensorDesc().setLayout(vpu::deviceLayout(outputLayout, layoutPreference));

        ASSERT_NO_THROW(_inferRequest.Infer());
        
        //
        // Check result
        //

        ref_scatter_elements_update(inputBlob, indicesBlob, updatesBlob, axis,
                                    referenceBlob);

    //  CompareCommonExact(outputBlob, referenceBlob); -- very inconvenient for debugging

        int errors = 0;

        const void* outputData = outputBlob->cbuffer();
        const void* referenceData = referenceBlob->cbuffer();

        const int outputSize = outputBlob->size();

        for (int i = 0; i < outputSize; i++) {
            double outputValue, referenceValue;

            if (precision == Precision::I32) {
                outputValue = reinterpret_cast<const int32_t*>(outputData)[i];
                referenceValue = reinterpret_cast<const int32_t*>(referenceData)[i];
            } else /* if (precision == Precision::FP16) */ {
                outputValue = PrecisionUtils::f16tof32(reinterpret_cast<const ie_fp16*>(outputData)[i]);
                referenceValue = PrecisionUtils::f16tof32(reinterpret_cast<const ie_fp16*>(referenceData)[i]);
            }

            if (outputValue != referenceValue) {
                if (errors++ < 25) {
                    std::cout << "error: index=" << index1DtoND(i, dataShape)
                              << ", outputValue=" << outputValue
                              << ", referenceValue=" << referenceValue
                              << std::endl;
                }
            }
        }

        ASSERT_EQ(errors, 0);
    }

private:

    static
    std::vector<int> index1DtoND(const int index1D,
                                 const std::vector<int>& shape) {
        int value = index1D;
        const int ndims = shape.size();
        std::vector<int> indexND(ndims);
        for (int i = ndims - 1; i >= 0; i--) {
            const int digit = value % shape[i];
                      value = value / shape[i];
            indexND[i] = digit;
        }
        return indexND;
    }

    // Count total number of elements in ND tensor
    static
    int getTotal(const std::vector<int>& shape) {
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
        } else if (Precision::I32 == precision) {
            const int ia = static_cast<int>(std::round(a));
            const int ib = static_cast<int>(std::round(b));
            std::uniform_int_distribution<int> uniform(ia, ib);
            for (int i = 0; i < num; i++) {
                const int v = uniform(gen);
                reinterpret_cast<int32_t*>(data)[i] = v;
            }
        } else {
            IE_ASSERT(precision == Precision::I32 ||
                      precision == Precision::FP16);
        }
    }

    // Note that:
    // - IR version is v7 (should be v10): as readNetwork() method
    //   cannot parse / denies IR v10 if there's no weights tensor
    static
    std::string createModel(const std::string     & dataType,
                            const std::vector<int>& dataShape,
                            const std::vector<int>& updatesShape) {
        std::string model = R"V0G0N(
            <?xml version="1.0" ?>
            <net name="testScatterElementsUpdate" version="7">
                <layers>
                    <layer id="0" name="input" type="Input">
                        <output>
                            <port id="0" precision="__TYPE__">
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
                            <port id="0" precision="__TYPE__">
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
                    <layer id="4" name="scatter" type="ScatterElementsUpdate">
                        <input>
                            <port id="0" precision="__TYPE__">
                                __INPUT_DIMS__
                            </port>
                            <port id="1" precision="I32">
                                __INDICES_DIMS__
                            </port>
                            <port id="2" precision="__TYPE__">
                                __UPDATES_DIMS__
                            </port>
                            <port id="3" precision="I32">
                            </port>
                        </input>
                        <output>
                            <port id="4" precision="__TYPE__">
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

        const std::string dataDimsStr = shapeToDimsString(dataShape);
        const std::string updatesDimsStr = shapeToDimsString(updatesShape);
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", dataDimsStr);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", dataDimsStr);
        REPLACE_WITH_STR(model, "__INDICES_DIMS__", updatesDimsStr);
        REPLACE_WITH_STR(model, "__UPDATES_DIMS__", updatesDimsStr);
        REPLACE_WITH_STR(model, "__TYPE__", dataType);

        return model;
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

private:
    std::mt19937 m_gen;
};

TEST_P(myriadLayersScatterElementsUpdateTest_smoke, accuracy) {
    testScatterElementsUpdate();
}
