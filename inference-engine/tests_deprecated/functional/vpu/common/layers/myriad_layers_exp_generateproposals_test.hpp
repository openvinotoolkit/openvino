// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"
#include "vpu/utils/error.hpp"

using namespace InferenceEngine;

#define NUM_COORDS (4)
#define ERROR_BOUND (2.5e-3f)

struct GenerateProposalsParam {
    float          min_size;
    float          nms_threshold;
    int            pre_nms_topn;
    int            post_nms_topn;


    friend std::ostream& operator<<(std::ostream& os, GenerateProposalsParam const& tst)
    {
        return os << "min size = " << tst.min_size
                  << ", nms threshold = " << tst.nms_threshold
                  << ", pre nms topn = " << tst.pre_nms_topn
                  << ", post nms topn = " << tst.post_nms_topn;
    };
};

using ExpGenerateProposalsTestParams = std::tuple<Dims, std::vector<int>, GenerateProposalsParam>;

typedef myriadLayerTestBaseWithParam<ExpGenerateProposalsTestParams> myriadLayersTestsExpGenerateProposals_smoke;

static void genInputs(InferenceEngine::BlobMap inputMap,
                      const int numProposals,
                      const int imgH, const int imgW) {
    const std::string INPUT_IM_INFO = "input0";
    const std::string INPUT_ANCHORS = "input1";
    const std::string INPUT_DELTAS  = "input2";
    const std::string INPUT_SCORES  = "input3";

    auto inputProposals = inputMap[INPUT_ANCHORS]->buffer().as<ie_fp16*>();
    auto inputDeltas    = inputMap[INPUT_DELTAS]->buffer().as<ie_fp16*>();
    auto inputScores    = inputMap[INPUT_SCORES]->buffer().as<ie_fp16*>();
    auto inputIMinfo    = inputMap[INPUT_IM_INFO]->buffer().as<ie_fp16*>();

    auto iScoresDims = inputMap[INPUT_SCORES]->getTensorDesc().getDims();

    // boxes generator
    auto genXY = [](int min, int max, int maxSize) {
            int a = min + maxSize * (static_cast<float>(rand()) / RAND_MAX);
            int b = a + maxSize * (static_cast<float>(rand()) / RAND_MAX) + 1;

            if (b > max) {
                const int d = b - max;
                a -= d;
                b -= d;
            }
            return std::make_pair(a, b);
        };

    // input boxes
    {
        const int X0 = 0, X1 = imgW, SX = (X1 - X0 + 1) * 4 / 5;
        const int Y0 = 0, Y1 = imgH, SY = (Y1 - Y0 + 1) * 4 / 5;

        for (int idx = 0; idx < numProposals; ++idx) {
            auto xx = genXY(X0, X1, SX);
            auto yy = genXY(Y0, Y1, SY);

            ie_fp16* iproposals = &inputProposals[idx * 4];

            iproposals[0] = PrecisionUtils::f32tof16( static_cast<float>(xx.first) );
            iproposals[1] = PrecisionUtils::f32tof16( static_cast<float>(yy.first) );
            iproposals[2] = PrecisionUtils::f32tof16( static_cast<float>(xx.second) );
            iproposals[3] = PrecisionUtils::f32tof16( static_cast<float>(yy.second) );
        }
    }

    const auto step_hw = iScoresDims[1] * iScoresDims[0];
    // input deltas
    for (int idx = 0; idx < iScoresDims[2]; ++idx) {
        for (int h = 0; h < iScoresDims[1]; ++h) {
            for (int w = 0; w < iScoresDims[0]; ++w) {
                const float maxDelta = 16.0f;
                float dx = maxDelta * (static_cast<float>(std::rand()) / RAND_MAX);
                float dy = maxDelta * (static_cast<float>(std::rand()) / RAND_MAX);

                const float maxlogDelta = 1000.f / 128;
                const float minlogDelta = 0.65;
                float d_log_w = std::log(minlogDelta + (maxlogDelta - minlogDelta) * (static_cast<float>(std::rand()) / RAND_MAX));
                float d_log_h = std::log(minlogDelta + (maxlogDelta - minlogDelta) * (static_cast<float>(std::rand()) / RAND_MAX));

                ie_fp16* ideltas = &inputDeltas[idx * step_hw * 4];

                ideltas[0 * step_hw] = PrecisionUtils::f32tof16( dx );
                ideltas[1 * step_hw] = PrecisionUtils::f32tof16( dy );
                ideltas[2 * step_hw] = PrecisionUtils::f32tof16( d_log_w );
                ideltas[3 * step_hw] = PrecisionUtils::f32tof16( d_log_h );
            }
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

        int count = inputMap[INPUT_SCORES]->size();

        int step = 0;
        for (auto p : primes) {
            if ((count % p) != 0) {
                step = p;
                break;
            }
        }
        IE_ASSERT(step != 0);

        ie_fp16 score = PrecisionUtils::f32tof16( 1.0f );
        ie_fp16 minScore = PrecisionUtils::f32tof16( 0.001f );
        int n = std::min(step/2, 1);
        for (int i = 0; i < count; ++i) {
            if ((uint32_t)score > (uint32_t)minScore)
                --score;

            inputScores[n] = score;
            n = (n + step) % count; // covers whole array since count & step are coprime ##s
        }
    }

    // image info
    inputIMinfo[0] = PrecisionUtils::f32tof16( (float) imgH );
    inputIMinfo[1] = PrecisionUtils::f32tof16( (float) imgW );
}

#ifdef __APPLE__
TEST_P(myriadLayersTestsExpGenerateProposals_smoke, DISABLED_ExpGenerateProposals) {
#else
TEST_P(myriadLayersTestsExpGenerateProposals_smoke, ExpGenerateProposals) {
#endif
    tensor_test_params scoresDims = std::get<0>(GetParam());
    std::vector<int> im_info = std::get<1>(GetParam());
    GenerateProposalsParam opParams = std::get<2>(GetParam());

    const auto numProposals = scoresDims.c * scoresDims.h * scoresDims.w;

    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    IN_OUT_desc inputTensors, outputTensors;
    inputTensors.push_back({3}); // im info
    inputTensors.push_back({numProposals, NUM_COORDS}); // input anchors
    inputTensors.push_back({scoresDims.c * NUM_COORDS,
                            scoresDims.h, scoresDims.w}); // input deltas
    inputTensors.push_back({scoresDims.c, scoresDims.h, scoresDims.w}); // input scores

    outputTensors.push_back({static_cast<size_t>(opParams.post_nms_topn), NUM_COORDS}); // output rois
    outputTensors.push_back({static_cast<size_t>(opParams.post_nms_topn)}); //output scores

    SetInputTensors(inputTensors);
    SetOutputTensors(outputTensors);

    std::map<std::string, std::string> layerParams = {
        {"min_size",          std::to_string(opParams.min_size)},
        {"nms_threshold",     std::to_string(opParams.nms_threshold)},
        {"post_nms_count",    std::to_string(opParams.post_nms_topn)},
        {"pre_nms_count",     std::to_string(opParams.pre_nms_topn)}
    };

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ExperimentalDetectronGenerateProposalsSingleImage").params(layerParams)));

    genInputs(_inputMap, numProposals, im_info.at(0), im_info.at(1));

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

    ref_ExpGenerateProposals(refInputBlobs,
                             refOutputBlobs,
                             opParams.min_size,
                             opParams.nms_threshold,
                             opParams.post_nms_topn,
                             opParams.pre_nms_topn);

    ASSERT_TRUE(Infer());

    int refIdx = 0;
    for (auto blob : _outputMap) {
        CompareCommonAbsolute(blob.second, refOutputBlobs[refIdx++], ERROR_BOUND);
    }
}

// Dimensions of scores input tensor
static std::vector<Dims> s_ExpGenerateProposalsLayerScores = {
    {
        Dims({1, 3, 8, 8}),
        Dims({1, 3, 15, 15}),
        Dims({1, 3, 30, 30}),
        Dims({1, 3, 60, 60}),
        Dims({1, 3, 120, 125}),
        Dims({1, 10, 240, 240}),
    },
};

static std::vector<std::vector<int>> s_ExpGenerateProposalsLayerImInfo = {
    {480, 480}, {240, 320}, {480, 320},
};

static std::vector<GenerateProposalsParam> s_ExpGenerateProposalsLayerParam = {
    {0.f, 0.7f, 100, 100}, {0.0f, 0.4f, 100, 100}, {0.0f, 0.9f, 100, 100}, {0.0f, 0.7f, 100, 50}, {4.0f, 0.7f, 100, 100},
};
