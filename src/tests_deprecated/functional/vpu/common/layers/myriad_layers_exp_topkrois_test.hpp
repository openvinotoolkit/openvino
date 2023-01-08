// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define NUM_COORDS (4)
#define ERROR_BOUND (2.5e-3f)

struct TopKROIsParam {
    int            max_rois;

    friend std::ostream& operator<<(std::ostream& os, TopKROIsParam const& tst)
    {
        return os << "max rois = " << tst.max_rois;
    };
};

using ExpTopKROIsTestParams = std::tuple<int, TopKROIsParam>;

typedef myriadLayerTestBaseWithParam<ExpTopKROIsTestParams> myriadLayersTestsExpTopKROIs_smoke;

static void genInputs(InferenceEngine::BlobMap inputMap) {
    const std::string INPUT_ROIS    = "input0";
    const std::string INPUT_SCORES  = "input1";

    const auto numRois = inputMap[INPUT_ROIS]->getTensorDesc().getDims()[0];

    auto inputRois   = inputMap[INPUT_ROIS]->buffer().as<ie_fp16*>();
    auto inputScores = inputMap[INPUT_SCORES]->buffer().as<ie_fp16*>();

    // boxes generator
    auto genXY = [](int min, int max, int maxSize) {
            int a = min + maxSize * (float(rand()) / float(RAND_MAX));
            int b = a + maxSize * (float(rand()) / float(RAND_MAX)) + 1;

            if (b > max) {
                const int d = b - max;
                a -= d;
                b -= d;
            }
            return std::make_pair(a, b);
        };

    // input boxes
    {
        const int minS = 200;
        const int maxS = 880;
        const int W = minS + maxS * (float(rand()) / float(RAND_MAX));
        const int H = minS + maxS * (float(rand()) / float(RAND_MAX));

        const int X0 = 0, X1 = W, SX = (X1 - X0 + 1) * 3 / 5;
        const int Y0 = 0, Y1 = H, SY = (Y1 - Y0 + 1) * 3 / 5;

        for (int idx = 0; idx < numRois; ++idx) {
            auto xx = genXY(X0, X1, SX);
            auto yy = genXY(Y0, Y1, SY);

            ie_fp16* irois = &inputRois[idx * 4];

            irois[0] = PrecisionUtils::f32tof16( (float) xx.first );
            irois[1] = PrecisionUtils::f32tof16( (float) yy.first );
            irois[2] = PrecisionUtils::f32tof16( (float) xx.second );
            irois[3] = PrecisionUtils::f32tof16( (float) yy.second );
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

        int step = 0;
        for (auto p : primes) {
            if ((numRois % p) != 0) {
                step = p;
                break;
            }
        }
        IE_ASSERT(step != 0); // unable to generate consistent scores list

        ie_fp16 score = PrecisionUtils::f32tof16( 1.0f );
        ie_fp16 minScore = PrecisionUtils::f32tof16( 0.001f );
        int n = std::min(step/2, 1);
        for (int i = 0; i < numRois; ++i) {
            if ((uint32_t)score > (uint32_t)minScore)
                --score;

            inputScores[n] = score;
            n = (n + step) % numRois; // covers whole array since numRois & step are coprime ##s
        }
    }
}

TEST_P(myriadLayersTestsExpTopKROIs_smoke, ExpTopKROIs) {
    int inputRoisNum = std::get<0>(GetParam());
    TopKROIsParam opParams = std::get<1>(GetParam());

    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    IN_OUT_desc inputTensors, outputTensors;
    inputTensors.push_back({static_cast<size_t>(inputRoisNum), NUM_COORDS}); // input rois
    inputTensors.push_back({static_cast<size_t>(inputRoisNum)}); // input probs

    outputTensors.push_back({static_cast<size_t>(opParams.max_rois), NUM_COORDS}); // output rois

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    std::map<std::string, std::string> layerParams = {
        {"max_rois", std::to_string(opParams.max_rois)},
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ExperimentalDetectronTopKROIs").params(layerParams)));

    genInputs(_inputMap);

    std::vector<InferenceEngine::Blob::Ptr> refInputBlobs;
    std::vector<InferenceEngine::Blob::Ptr> refOutputBlobs;

    for (const auto& blob : _inputMap) {
        auto _refInputBlob = make_shared_blob<ie_fp16>({Precision::FP16,
                                                        blob.second->getTensorDesc().getDims(),
                                                        blob.second->getTensorDesc().getLayout()},
                                                        blob.second->buffer());
        refInputBlobs.push_back(_refInputBlob);
    }

    for (const auto& blob : _outputMap) {
        auto refOutputBlob = make_shared_blob<ie_fp16>({Precision::FP16,
                                                      blob.second->getTensorDesc().getDims(),
                                                      blob.second->getTensorDesc().getLayout()});
        refOutputBlob->allocate();
        refOutputBlobs.push_back(refOutputBlob);
    }

    ref_ExpTopKROIs(refInputBlobs,
                    refOutputBlobs,
                    opParams.max_rois);

    ASSERT_TRUE(Infer());

    CompareCommonAbsolute(_outputMap.begin()->second, refOutputBlobs[0], ERROR_BOUND);
}

static std::vector<int> s_ExpTopKROIsInputRoisNum = {
    100, 150, 50, 200, 101
};

static std::vector<TopKROIsParam> s_ExpTopKROIsMaxRoisNum = {
    { 100 }, { 150 }, { 50 }, { 101 }
};
