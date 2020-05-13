// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "vpu_tests_config.hpp"
#include "vpu_case_common.hpp"

#include <algorithm>
#include <random>
#include <vector>
#include <string>

using namespace InferenceEngine;

using   InputShape = std::vector<int>;
using IndicesShape = std::vector<int>;
using         Axis = int;
using         Type = std::string;  // "FP16", "I32"

using GatherTestParams = std::tuple<InputShape,
                                    IndicesShape,
                                    Axis,
                                    Type>;

class myriadLayerGather_smoke :
    public myriadLayerTestBaseWithParam<GatherTestParams> {
protected:

    void testGather() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

        //
        // Parse and check test parameters
        //

        const GatherTestParams& gatherTestParams = GetParam();
        const std::vector<int>&   inputShape = std::get<0>(gatherTestParams);
        const std::vector<int>& indicesShape = std::get<1>(gatherTestParams);
        const             int      axisParam = std::get<2>(gatherTestParams);
        const std::string     &         type = std::get<3>(gatherTestParams);

        IE_ASSERT(type == "I32" ||
                  type == "FP16");

        const int indicesNDims = indicesShape.size();
        const int   inputNDims =   inputShape.size();
        const int  outputNDims = indicesNDims + inputNDims - 1;
        IE_ASSERT(outputNDims > 0);

        // NB: axis param must be in [-len(in.shape), len(in.shape)-1]
        const int axis = axisParam + (axisParam < 0 ? inputNDims : 0);
        IE_ASSERT(0 <= axis && axis < inputNDims);

        // Deduce shape of `output` tensor
        //
        // E.g.:
        //    {N, C, H, W} could be shape of `input`
        // {I, J}          could be shape of `indices`
        // {I, J, C, H, W} could be shape of `output`
        std::vector<int> outputShape;
        for (int i = 0; i < axis; i++) {
            outputShape.push_back(inputShape[i]);
        }
        for (int i = 0; i < indicesNDims; i++) {
            outputShape.push_back(indicesShape[i]);
        }
        for (int i = axis + 1; i < inputNDims; i++) {
            outputShape.push_back(inputShape[i]);
        }
        IE_ASSERT(outputShape.size() == outputNDims);

        //
        // Skip test if data is too large for device
        //

        const int inputTotal = getTotal(inputShape);
        const int outputTotal = getTotal(outputShape);
        const int indicesTotal = getTotal(indicesShape);

        const Precision precision = type == "I32" ?
                                        Precision::I32 :
                                        Precision::FP16;

        const int bpp = precision == Precision::I32 ?
                                         sizeof(int32_t) :
                                         sizeof(ie_fp16);

        const int threshold = 50 * (1 << 20);  // empirical

        const bool tooLarge = inputTotal * bpp > threshold ||
                             outputTotal * bpp > threshold;

        DISABLE_IF(tooLarge && !CheckMA2085());

        //
        // Initialize 1-layer network
        //

        std::string model = createModel(inputShape,
                                        outputShape,
                                        indicesShape,
                                        axis,
                                        type);

        ASSERT_NO_THROW(readNetwork(model));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(precision);
        _inputsInfo["indices"]->setPrecision(Precision::I32);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["gather"]->setPrecision(precision);

        //
        // Create infer request and get its blobs pointers
        //

        StatusCode st = OK;

        ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, _config, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("input", inputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr indicesBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("indices", indicesBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("gather", outputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr referenceBlob;
        if (type == "I32") {
            referenceBlob = make_shared_blob<int32_t>(outputBlob->getTensorDesc());
        } else {
            referenceBlob = make_shared_blob<ie_fp16>(outputBlob->getTensorDesc());
        }
        referenceBlob->allocate();

        //
        // Initialize `input` and `indices` blobs
        //

        void* inputBlobData = inputBlob->buffer();
        ASSERT_NE(inputBlobData, nullptr);

        void* indicesBlobData = indicesBlob->buffer();
        ASSERT_NE(indicesBlobData, nullptr);

        const int indicesLimit = inputShape[axis] - 1;

        std::mt19937 gen;
        fillUniformly(inputBlobData, inputTotal, precision, 0, 255, gen);
        fillUniformly(indicesBlobData, indicesTotal, Precision::I32, 0, indicesLimit, gen);

        //
        // Infer
        //

        const auto inputLayout = inputBlob->getTensorDesc().getLayout();
        const auto outputLayout = outputBlob->getTensorDesc().getLayout();
        const auto indicesLayout = indicesBlob->getTensorDesc().getLayout();
        const auto layoutPreference = vpu::LayoutPreference::ChannelMajor;

        inputBlob->getTensorDesc().setLayout(vpu::deviceLayout(inputLayout, layoutPreference));
        indicesBlob->getTensorDesc().setLayout(vpu::deviceLayout(indicesLayout, layoutPreference));
        outputBlob->getTensorDesc().setLayout(vpu::deviceLayout(outputLayout, layoutPreference));
        referenceBlob->getTensorDesc().setLayout(vpu::deviceLayout(outputLayout, layoutPreference));

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        //
        // Check result
        //

        ref_gather(indicesBlob, inputBlob, referenceBlob, axis);

        CompareCommonExact(outputBlob, referenceBlob);
    }

private:

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
    std::string createModel(const std::vector<int>& inputShape,
                            const std::vector<int>& outputShape,
                            const std::vector<int>& indicesShape,
                            const             int   axis,
                            const std::string     & type) {
        std::string model = R"V0G0N(
            <?xml version="1.0" ?>
            <net name="testGather" version="7">
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
                    <layer id="2" name="gather" type="Gather">
                        <data axis="__AXIS__"/>
                        <input>
                            <port id="0" precision="__TYPE__">
                                __INPUT_DIMS__
                            </port>
                            <port id="1" precision="I32">
                                __INDICES_DIMS__
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
                    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
                </edges>
            </net>
        )V0G0N";

        const std::string inputDimsStr = shapeToDimsString(inputShape);
        const std::string outputDimsStr = shapeToDimsString(outputShape);
        const std::string indicesDimsStr = shapeToDimsString(indicesShape);
        const std::string axisStr = std::to_string(axis);
        REPLACE_WITH_STR(model, "__INPUT_DIMS__", inputDimsStr);
        REPLACE_WITH_STR(model, "__OUTPUT_DIMS__", outputDimsStr);
        REPLACE_WITH_STR(model, "__INDICES_DIMS__", indicesDimsStr);
        REPLACE_WITH_STR(model, "__AXIS__", axisStr);
        REPLACE_WITH_STR(model, "__TYPE__", type);

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
};

TEST_P(myriadLayerGather_smoke, Gather) {
    testGather();
}
