// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

using namespace InferenceEngine;

typedef std::vector<int> NMS_Dims;
typedef std::vector<std::vector<std::vector<float>>> init3DFloat;
typedef std::vector<int> initIntScalar;
typedef std::vector<float> initFPScalar;
typedef std::vector<std::vector<int>> refType;
struct NMS_testParams {
    int dims[3]; // {spat_dim, num_classes, num_batches}
    int centerPointBox;
    initIntScalar MaxOutBoxesPerClass; // scalar
    initFPScalar IoUThreshold; // scalar
    initFPScalar ScoreThreshold; // scalar
    init3DFloat boxes;
    init3DFloat scores;
    refType referenceOutput;
};
static std::string getModel(const int numOfInputs, const NMS_Dims &dims, const int center_point_box) {
    std::string model = R"V0G0N(
                <net name="testNMS" version="7">
                    <layers>
                        <layer id="0" name="boxes" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>__BATCHES__</dim>
                                    <dim>__SPAT_DIM__</dim>
                                    <dim>4</dim>
                                </port>
                            </output>
                        </layer>
                        <layer id="1" name="scores" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>__BATCHES__</dim>
                                    <dim>__CLASSES__</dim>
                                    <dim>__SPAT_DIM__</dim>
                                </port>
                            </output>
                        </layer>)V0G0N";
    if (numOfInputs > 2)
        model += R"V0G0N(
                        <layer id="2" name="MaxOutputBoxesPerClass" precision="I32" type="Input">
                            <output>
                                <port id="0">
                                    <dim>1</dim>
                                </port>
                            </output>
                        </layer>)V0G0N";
    if (numOfInputs > 3)
        model += R"V0G0N(
                        <layer id="3" name="IoUThreshold" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>1</dim>
                                </port>
                            </output>
                        </layer>)V0G0N";
    if (numOfInputs > 4)
        model += R"V0G0N(
                        <layer id="4" name="ScoreThreshold" precision="FP16" type="Input">
                            <output>
                                <port id="0">
                                    <dim>1</dim>
                                </port>
                            </output>
                        </layer>)V0G0N";
    model += R"V0G0N(
                        <layer id="5" name="NMS" precision="I32" type="NonMaxSuppression">
                            <data center_point_box="__CPB__"/>
                            <input>
                                <port id="0">
                                    <dim>__BATCHES__</dim>
                                    <dim>__SPAT_DIM__</dim>
                                    <dim>4</dim>
                                </port>
                                <port id="1">
                                    <dim>__BATCHES__</dim>
                                    <dim>__CLASSES__</dim>
                                    <dim>__SPAT_DIM__</dim>
                                </port>)V0G0N";
    if (numOfInputs > 2)
        model += R"V0G0N(
                                <port id="2">
                                    <dim>1</dim>
                                </port>)V0G0N";
    if (numOfInputs > 3)
        model += R"V0G0N(
                                <port id="3">
                                    <dim>1</dim>
                                </port>)V0G0N";
    if (numOfInputs > 4)
        model += R"V0G0N(
                                <port id="4">
                                    <dim>1</dim>
                                </port>)V0G0N";
    model += R"V0G0N(
                            </input>
                            <output>
                                <port id="4">
                                    <dim>__SPAT_DIM__</dim>
                                    <dim>3</dim>
                                </port>
                            </output>
                        </layer>
                    </layers>
                    <edges>
                        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
                        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>)V0G0N";
    if (numOfInputs > 2)
        model += R"V0G0N(
                        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>)V0G0N";
    if (numOfInputs > 3)
        model += R"V0G0N(
                        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>)V0G0N";
    if (numOfInputs > 4)
        model += R"V0G0N(
                        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>)V0G0N";
    model += R"V0G0N(
                    </edges>
                </net>
            )V0G0N";

    REPLACE_WITH_STR(model, "__SPAT_DIM__", std::to_string(dims[0]));
    REPLACE_WITH_STR(model, "__CLASSES__", std::to_string(dims[1]));
    REPLACE_WITH_STR(model, "__BATCHES__", std::to_string(dims[2]));
    REPLACE_WITH_STR(model, "__CPB__", std::to_string(center_point_box));

    return model;
}

static void copyScalarToBlob(const Blob::Ptr& blob, const initIntScalar& scalar) {
    auto *data = blob->buffer().as<int32_t *>();
    data[0] = scalar[0];
}

static void copyScalarToBlob(const Blob::Ptr& blob, const initFPScalar& scalar) {
    auto *data = blob->buffer().as<ie_fp16 *>();
    data[0] = PrecisionUtils::f32tof16(scalar[0]);
}

static void copy3DToBlob(const Blob::Ptr& blob, const init3DFloat& src) {
    auto *data = blob->buffer().as<ie_fp16 *>();
    const auto dims = blob->getTensorDesc().getDims();
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                data[i * dims[1] * dims[2] + j * dims[2] + k] = PrecisionUtils::f32tof16(src[i][j][k]);
            }
        }
    }
}

static void copyReference(const Blob::Ptr& blob, const refType src) {
    int32_t *data = blob->buffer().as<int32_t *>();
    const auto dims = blob->getTensorDesc().getDims();

    int boxNum = 0;
    for (; boxNum < src.size(); boxNum++) {
        data[boxNum * 3 + 0] = src[boxNum][0];
        data[boxNum * 3 + 1] = src[boxNum][1];
        data[boxNum * 3 + 2] = src[boxNum][2];
    }
    for (; boxNum < dims[0]; boxNum++) {
        data[boxNum * 3 + 0] = -1;
        data[boxNum * 3 + 1] = -1;
        data[boxNum * 3 + 2] = -1;
    }
}

typedef myriadLayerTestBaseWithParam<NMS_testParams> myriadLayersTestsNonMaxSuppression_smoke;

TEST_P(myriadLayersTestsNonMaxSuppression_smoke, NonMaxSuppression) {
    const auto params = GetParam();
    const int spatDim = params.dims[0];
    const int numClasses = params.dims[1];
    const int numBatches = params.dims[2];
    const int center_point_box = params.centerPointBox;

    int numOfInputs = 2;
    if (!params.ScoreThreshold.empty())
        numOfInputs = 5;
    else if (!params.IoUThreshold.empty())
        numOfInputs = 4;
    else if (!params.MaxOutBoxesPerClass.empty())
        numOfInputs = 3;

    const auto model = getModel(numOfInputs, {spatDim, numClasses, numBatches}, center_point_box);
    ASSERT_NO_THROW(readNetwork(model));

    const auto& network = _cnnNetwork;
    _inputsInfo = network.getInputsInfo();
    _inputsInfo["boxes"]->setPrecision(Precision::FP16);
    _inputsInfo["scores"]->setPrecision(Precision::FP16);
    if (numOfInputs > 2)
        _inputsInfo["MaxOutputBoxesPerClass"]->setPrecision(Precision::I32);
    if (numOfInputs > 3)
        _inputsInfo["IoUThreshold"]->setPrecision(Precision::FP16);
    if (numOfInputs > 4)
        _inputsInfo["ScoreThreshold"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["NMS"]->setPrecision(Precision::I32);

    StatusCode st = OK;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, _config, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr boxesBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("boxes", boxesBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    std::cout << CheckMyriadX() << std::endl;
    copy3DToBlob(boxesBlob, params.boxes);

    Blob::Ptr scoresBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("scores", scoresBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    copy3DToBlob(scoresBlob, params.scores);

    if (numOfInputs > 2) {
        Blob::Ptr MaxOutputBoxesBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("MaxOutputBoxesPerClass", MaxOutputBoxesBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        copyScalarToBlob(MaxOutputBoxesBlob, params.MaxOutBoxesPerClass);
    }

    if (numOfInputs > 3) {
        Blob::Ptr IoUThresholdBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("IoUThreshold", IoUThresholdBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        copyScalarToBlob(IoUThresholdBlob, params.IoUThreshold);
    }

    if (numOfInputs > 4) {
        Blob::Ptr ScoreThresholdBlob;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("ScoreThreshold", ScoreThresholdBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        copyScalarToBlob(ScoreThresholdBlob, params.ScoreThreshold);
    }

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("NMS", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr refBlob = make_shared_blob<int32_t>(outputBlob->getTensorDesc());
    refBlob->allocate();
    copyReference(refBlob, params.referenceOutput);

    if (memcmp(refBlob->cbuffer(), outputBlob->cbuffer(), outputBlob->byteSize()))
        FAIL() << "Wrong result with compare ONNX reference!";
}
