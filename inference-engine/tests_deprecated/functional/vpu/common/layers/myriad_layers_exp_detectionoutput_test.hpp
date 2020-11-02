// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"
#include "tests_vpu_common.hpp"

#include <algorithm>
#include <functional>
#include <string>

using namespace InferenceEngine;

struct SizeParams {
    int numRois;
    int numClasses;
    int maxDetections;
};

static void generateData(Blob::Ptr inputBoxesBlob,
                         Blob::Ptr inputDeltasBlob,
                         Blob::Ptr inputScoresBlob,
                         Blob::Ptr inputIMinfoBlob,
                         const SizeParams& sizeParams,
                         const ExpDetectionOutputParams& layerParams)
{
    auto inputBoxes  = inputBoxesBlob->buffer().as<ie_fp16*>();
    auto inputDeltas = inputDeltasBlob->buffer().as<ie_fp16*>();
    auto inputScores = inputScoresBlob->buffer().as<ie_fp16*>();
    auto inputIMinfo = inputIMinfoBlob->buffer().as<ie_fp16*>();

    const size_t numRois    = sizeParams.numRois;
    const size_t numClasses = sizeParams.numClasses;

    const int W = 320;
    const int H = 240;

    // boxes generator
    auto genXY = [](int min, int max, int minSize, int maxSize)
        {
            int a = min + maxSize * (float(std::rand()) / RAND_MAX);
            int b = min + maxSize * (float(std::rand()) / RAND_MAX);
            if (b < a)
                std::swap(a, b);
            if (b - a < minSize)
                b = a + minSize;
            if (b > max)
            {
                const int d = b - max;
                a -= d;
                b -= d;
            }
            return std::make_pair(a, b);
        };

    // input boxes
    {
        const int DX = 5 * layerParams.deltas_weights[0];
        const int DY = 5 * layerParams.deltas_weights[1];

        const int X0 = 0 + DX, X1 = W - DX, SX = X1 - X0 + 1;
        const int Y0 = 0 + DY, Y1 = W - DY, SY = Y1 - Y0 + 1;

        for (int roi_idx = 0; roi_idx < numRois; ++roi_idx)
        {
            auto xx = genXY(X0, X1, DX, SX);
            auto yy = genXY(X0, X1, DY, SY);

            ie_fp16* iboxes = &inputBoxes[roi_idx * 4];

            iboxes[0] = PrecisionUtils::f32tof16( (float) xx.first );
            iboxes[1] = PrecisionUtils::f32tof16( (float) yy.first );
            iboxes[2] = PrecisionUtils::f32tof16( (float) xx.second );
            iboxes[3] = PrecisionUtils::f32tof16( (float) yy.second );
        }
    }

    // input deltas
    for (int roi_idx = 0; roi_idx < numRois; ++roi_idx)
    {
        for (int class_idx = 0; class_idx < numClasses; ++class_idx)
        {
            float dx = 0.5*layerParams.deltas_weights[0] + layerParams.deltas_weights[0] * (float(std::rand()) / RAND_MAX);
            float dy = 0.5*layerParams.deltas_weights[1] + layerParams.deltas_weights[1] * (float(std::rand()) / RAND_MAX);

            const float minD = 0.95;
            const float maxD = 1.10;

            float d_log_w = std::log(layerParams.deltas_weights[2] * (minD + (maxD - minD) * (float(std::rand()) / RAND_MAX)));
            float d_log_h = std::log(layerParams.deltas_weights[3] * (minD + (maxD - minD) * (float(std::rand()) / RAND_MAX)));

            ie_fp16* ideltas = &inputDeltas[(roi_idx * numClasses + class_idx) * 4];

            ideltas[0] = PrecisionUtils::f32tof16( dx );
            ideltas[1] = PrecisionUtils::f32tof16( dy );
            ideltas[2] = PrecisionUtils::f32tof16( d_log_w );
            ideltas[3] = PrecisionUtils::f32tof16( d_log_h );
        }
    }

    // input scores
    // for the stable testing reasons, we try to produce totally different scores
    // fp16 has 2^16 different codes (including nans, etc), but we have to generate at least 81000 (81*1000),
    // so we use all successive FP numbers, starting from 1.0-1ulp towards 0, until small value is reached
    // (less than score_threshold), so all such small score values can be the same
    // score tensor is filled in random-like manner by using index step which is coprime with overall size
    {
        static const int primes[] = {97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47, 43,
                                     41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2};

        int count = numRois * numClasses;

        int step = 0;
        for (auto p : primes)
        {
            if ((count % p) != 0)
            {
                step = p;
                break;
            }
        }
        IE_ASSERT(step != 0); // unable to generate consistent scores list

        ie_fp16 score = PrecisionUtils::f32tof16( 1.0f );
        ie_fp16 minScore = PrecisionUtils::f32tof16( 0.001f );
        int n = std::min(step/2, 1);
        for (int i = 0; i < count; ++i)
        {
            if ((uint32_t)score > (uint32_t)minScore)
                --score;
            inputScores[n] = score;
            n = (n + step) % count; // covers whole array since count & step are coprime ##s
        }
    }

    // image info
    inputIMinfo[0] = PrecisionUtils::f32tof16( (float) H );
    inputIMinfo[1] = PrecisionUtils::f32tof16( (float) W );
}

using ExpDetectionOutputTestParams = std::tuple<SizeParams, ExpDetectionOutputParams>;

static const Precision dataPrecision = Precision::FP16;
static const Precision classPrecision = Precision::I32;

enum BlobIndices { InputBoxes=0, InputDeltas, InputScores, InputIMinfo,
                   OutputBoxes, OutputClasses, OutputScores, NumBlobs };

typedef std::vector<SizeVector> BlobDimsList;

class ExpDetectionOutputTest: public myriadLayerTestBaseWithParam<ExpDetectionOutputTestParams>
{
protected:
    void testExpDetectionOutput()
        {
            _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

            const auto testParams = GetParam();
            const auto sizeParams = std::get<0>(testParams);
            const auto layerParams = std::get<1>(testParams);

            const auto blobDims = calcBlobDims(sizeParams);

            const auto model = getModel(blobDims, layerParams);

            ASSERT_NO_THROW(readNetwork(model));

            const auto& network = _cnnNetwork;

            _inputsInfo = network.getInputsInfo();
            _inputsInfo["detectionOutput_inputBoxes"]->setPrecision(dataPrecision);
            _inputsInfo["detectionOutput_inputBoxes"]->setLayout(defaultLayout(blobDims[InputBoxes].size()));
            _inputsInfo["detectionOutput_inputDeltas"]->setPrecision(dataPrecision);
            _inputsInfo["detectionOutput_inputDeltas"]->setLayout(defaultLayout(blobDims[InputDeltas].size()));
            _inputsInfo["detectionOutput_inputScores"]->setPrecision(dataPrecision);
            _inputsInfo["detectionOutput_inputScores"]->setLayout(defaultLayout(blobDims[InputScores].size()));
            _inputsInfo["detectionOutput_inputIMinfo"]->setPrecision(dataPrecision);
            _inputsInfo["detectionOutput_inputIMinfo"]->setLayout(defaultLayout(blobDims[InputIMinfo].size()));

            _outputsInfo = network.getOutputsInfo();
            _outputsInfo["expDetectionOutput.0"]->setPrecision(dataPrecision);
            _outputsInfo["expDetectionOutput.0"]->setLayout(defaultLayout(blobDims[OutputBoxes].size()));
            _outputsInfo["expDetectionOutput.1"]->setPrecision(classPrecision);
            _outputsInfo["expDetectionOutput.1"]->setLayout(defaultLayout(blobDims[OutputClasses].size()));
            _outputsInfo["expDetectionOutput.2"]->setPrecision(dataPrecision);
            _outputsInfo["expDetectionOutput.2"]->setLayout(defaultLayout(blobDims[OutputScores].size()));

            StatusCode st = OK;

            ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(_exeNetwork, network, _config, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

            ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            Blob::Ptr inputBoxesBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("detectionOutput_inputBoxes", inputBoxesBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            Blob::Ptr inputDeltasBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("detectionOutput_inputDeltas", inputDeltasBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            Blob::Ptr inputScoresBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("detectionOutput_inputScores", inputScoresBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            Blob::Ptr inputIMinfoBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("detectionOutput_inputIMinfo", inputIMinfoBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            generateData(inputBoxesBlob, inputDeltasBlob, inputScoresBlob, inputIMinfoBlob, sizeParams, layerParams);

            ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            Blob::Ptr outputBoxesBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("expDetectionOutput.0", outputBoxesBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            Blob::Ptr refBoxesBlob = make_shared_blob<ie_fp16>(outputBoxesBlob->getTensorDesc());
            refBoxesBlob->allocate();

            Blob::Ptr outputClassesBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("expDetectionOutput.1", outputClassesBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            Blob::Ptr refClassesBlob = make_shared_blob<int32_t>(outputClassesBlob->getTensorDesc());
            refClassesBlob->allocate();

            Blob::Ptr outputScoresBlob;
            ASSERT_NO_THROW(st = _inferRequest->GetBlob("expDetectionOutput.2", outputScoresBlob, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            Blob::Ptr refScoresBlob = make_shared_blob<ie_fp16>(outputScoresBlob->getTensorDesc());
            refScoresBlob->allocate();

            ref_expDetectionOutput(inputBoxesBlob, inputDeltasBlob, inputScoresBlob, inputIMinfoBlob,
                                   refBoxesBlob, refClassesBlob, refScoresBlob,
                                   sizeParams.numRois, sizeParams.numClasses, sizeParams.maxDetections, layerParams);

            CompareCommonAbsolute(refBoxesBlob, outputBoxesBlob, 0.0f);
            CompareCommonExact(refClassesBlob, outputClassesBlob);
            CompareCommonAbsolute(refScoresBlob, outputScoresBlob, 0.0f);
        }
    static std::string getModel(const BlobDimsList& blobDims, const ExpDetectionOutputParams& layerParams)
        {
            std::string model = R"V0G0N(
                <net name="testExpDetectionOutput" version="5">
                    <layers>
                        <layer id="0" name="detectionOutput_inputBoxes" type="Input">
                            <output>
                                <port id="0" precision="__DATA_PRECISION__">__INPUT_BOXES_DIMS__</port>
                            </output>
                        </layer>
                        <layer id="1" name="detectionOutput_inputDeltas" type="Input">
                            <output>
                                <port id="0" precision="__DATA_PRECISION__">__INPUT_DELTAS_DIMS__</port>
                            </output>
                        </layer>
                        <layer id="2" name="detectionOutput_inputScores" type="Input">
                            <output>
                                <port id="0" precision="__DATA_PRECISION__">__INPUT_SCORES_DIMS__</port>
                            </output>
                        </layer>
                        <layer id="3" name="detectionOutput_inputIMinfo" type="Input">
                            <output>
                                <port id="0" precision="__DATA_PRECISION__">__INPUT_IM_INFO_DIMS__</port>
                            </output>
                        </layer>
                        <layer id="4" name="expDetectionOutput" type="ExperimentalDetectronDetectionOutput">
                             <data __LAYER_PARAMS__/>
                             <input>
                                 <port id="0">__INPUT_BOXES_DIMS__</port>
                                 <port id="1">__INPUT_DELTAS_DIMS__</port>
                                 <port id="2">__INPUT_SCORES_DIMS__</port>
                                 <port id="3">__INPUT_IM_INFO_DIMS__</port>
                             </input>
                             <output>
                                 <port id="4" precision="__DATA_PRECISION__">__OUTPUT_BOXES_DIMS__</port>
                                 <port id="5" precision="__CLASS_PRECISION__">__OUTPUT_CLASSES_DIMS__</port>
                                 <port id="6" precision="__DATA_PRECISION__">__OUTPUT_SCORES_DIMS__</port>
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

            const auto inputBoxesDimsStr = dimsToString(blobDims[InputBoxes]);
            const auto inputDeltasDimsStr = dimsToString(blobDims[InputDeltas]);
            const auto inputScoresDimsStr = dimsToString(blobDims[InputScores]);
            const auto inputIMinfoDimsStr = dimsToString(blobDims[InputIMinfo]);

            const auto outputBoxesDimsStr = dimsToString(blobDims[OutputBoxes]);
            const auto outputClassesDimsStr = dimsToString(blobDims[OutputClasses]);
            const auto outputScoresDimsStr = dimsToString(blobDims[OutputScores]);

            const auto layerParamsStr = layerParamsToString(layerParams);

            REPLACE_WITH_STR(model, "__DATA_PRECISION__", dataPrecision.name());
            REPLACE_WITH_STR(model, "__CLASS_PRECISION__", classPrecision.name());

            REPLACE_WITH_STR(model, "__INPUT_BOXES_DIMS__", inputBoxesDimsStr);
            REPLACE_WITH_STR(model, "__INPUT_DELTAS_DIMS__", inputDeltasDimsStr);
            REPLACE_WITH_STR(model, "__INPUT_SCORES_DIMS__", inputScoresDimsStr);
            REPLACE_WITH_STR(model, "__INPUT_IM_INFO_DIMS__", inputIMinfoDimsStr);

            REPLACE_WITH_STR(model, "__OUTPUT_BOXES_DIMS__", outputBoxesDimsStr);
            REPLACE_WITH_STR(model, "__OUTPUT_CLASSES_DIMS__", outputClassesDimsStr);
            REPLACE_WITH_STR(model, "__OUTPUT_SCORES_DIMS__", outputScoresDimsStr);

            REPLACE_WITH_STR(model, "__LAYER_PARAMS__", layerParamsStr);

            return model;
        }
    static std::string layerParamsToString(const ExpDetectionOutputParams& layerParams)
        {
            std::string str;

            str += "deltas_weights=\"";
            const char* sep = "";
            for (auto& w : layerParams.deltas_weights)
            {
                str += sep + std::to_string(w);
                sep = ",";
            }
            str += "\"";

            str += " max_delta_log_wh=\"" + std::to_string(layerParams.max_delta_log_wh) + "\"";
            str += " nms_threshold=\"" + std::to_string(layerParams.nms_threshold) + "\"";
            str += " score_threshold=\"" + std::to_string(layerParams.score_threshold) + "\"";
            str += " max_detections_per_image=\"" + std::to_string(layerParams.max_detections_per_image) + "\"";
            str += " num_classes=\"" + std::to_string(layerParams.num_classes) + "\"";
            str += " post_nms_count=\"" + std::to_string(layerParams.post_nms_count) + "\"";
            str += " class_agnostic_box_regression=\"" + std::to_string(layerParams.class_agnostic_box_regression) + "\"";

            return str;
        }
    static std::string dimsToString(const SizeVector& dims)
        {
            std::string str;
            for (auto& d : dims)
                str += "<dim>" + std::to_string(d) + "</dim>";
            return str;
        }
    static BlobDimsList calcBlobDims(const SizeParams& sizeParams)
        {
            const size_t numRois       = sizeParams.numRois;
            const size_t numClasses    = sizeParams.numClasses;
            const size_t maxDetections = sizeParams.maxDetections;

            BlobDimsList list(NumBlobs);

            list[InputBoxes]    = SizeVector({numRois, 4});
            list[InputDeltas]   = SizeVector({numRois, numClasses * 4});
            list[InputScores]   = SizeVector({numRois, numClasses});
            list[InputIMinfo]   = SizeVector({1, 3});

            list[OutputBoxes]   = SizeVector({maxDetections, 4});
            list[OutputClasses] = SizeVector({maxDetections});
            list[OutputScores]  = SizeVector({maxDetections});

            return list;
        }
    static Layout defaultLayout(int ndims)
        {
            switch (ndims)
            {
            case 5: return NCDHW;
            case 4: return NCHW;
            case 3: return CHW;
            case 2: return NC;
            case 1: return C;
            }
            return ANY;
        }
};

class myriadTestsExpDetectionOutput_smoke: public ExpDetectionOutputTest
{
};

TEST_P(myriadTestsExpDetectionOutput_smoke, ExpDetectionOutput)
{
    testExpDetectionOutput();
}
