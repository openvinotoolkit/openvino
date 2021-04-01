// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_reference_functions.hpp"
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define NUM_ELEMS_PRIORS (4)
#define ERROR_BOUND (2.5e-3f)

extern const char ALL_INPUTS[] = "all_inputs";
extern const char MISSING_FEATURE_MAP[] = "no_feature_map";
extern const char MISSING_INPUT_IMAGE[] = "no_input_image";

struct PriorGridGeneratorParam {
    int               flatten;
    int               grid_w;
    int               grid_h;
    float             stride_w;
    float             stride_h;

    friend std::ostream& operator<<(std::ostream& os, PriorGridGeneratorParam const& tst)
    {
        return os << "grid width = " << tst.grid_w
                  << ", grid height = " << tst.grid_h
                  << ", step width = " << tst.stride_w
                  << ", step height = " << tst.stride_h;
    };
};

struct InputDims {
    tensor_test_params priors;
    tensor_test_params featureMap;
    tensor_test_params imData;

    InputDims(Dims priorDims, Dims featureMapDims, Dims imDataDims) :
                priors(priorDims),
                featureMap(featureMapDims),
                imData(imDataDims) {}
    
    InputDims() = default;
};

using ExpPriorGridGeneratorTestParams = std::tuple<InputDims, PriorGridGeneratorParam>;

template <const char* InputsAvailable>
class myriadLayersTestsExpPriorGridGen : public myriadLayerTestBaseWithParam<ExpPriorGridGeneratorTestParams> {
    protected:
        void genPriors(InferenceEngine::Blob::Ptr rois,
                       const int width,
                       const int height,
                       const uint32_t numPriors) {
            ie_fp16 *roisBlobData = rois->buffer().as<ie_fp16*>();
            const int maxRangeWidth  = width * 4 / 5;
            const int maxRangeHeight = height * 4 / 5;

            for (int i = 0; i < numPriors; i++) {
                int x0 = std::rand() % maxRangeWidth;
                int x1 = x0 + (std::rand() % (width - x0 - 1)) + 1;
                int y0 = std::rand() % maxRangeHeight;
                int y1 = y0 + (std::rand() % (height - y0 - 1)) + 1;

                roisBlobData[i * NUM_ELEMS_PRIORS + 0] = PrecisionUtils::f32tof16((float)x0);
                roisBlobData[i * NUM_ELEMS_PRIORS + 1] = PrecisionUtils::f32tof16((float)y0);
                roisBlobData[i * NUM_ELEMS_PRIORS + 2] = PrecisionUtils::f32tof16((float)x1);
                roisBlobData[i * NUM_ELEMS_PRIORS + 3] = PrecisionUtils::f32tof16((float)y1);
            }
        }

        void runTest() {
            InputDims inputTensorsDims = std::get<0>(GetParam());
            PriorGridGeneratorParam opParams = std::get<1>(GetParam());

            const auto numPriors = inputTensorsDims.priors.n;

            _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

            IN_OUT_desc inputTensors, outputTensors;
            inputTensors.push_back({inputTensorsDims.priors.n, inputTensorsDims.priors.c});
            inputTensors.push_back({inputTensorsDims.featureMap.n,
                                    inputTensorsDims.featureMap.c,
                                    inputTensorsDims.featureMap.h,
                                    inputTensorsDims.featureMap.w});
            inputTensors.push_back({inputTensorsDims.imData.n,
                                    inputTensorsDims.imData.c,
                                    inputTensorsDims.imData.h,
                                    inputTensorsDims.imData.w});

            const int gridWidth  = opParams.grid_w ? opParams.grid_w : inputTensorsDims.featureMap.w;
            const int gridHeight = opParams.grid_h ? opParams.grid_h : inputTensorsDims.featureMap.h;

            outputTensors.push_back({numPriors * gridHeight * gridWidth, inputTensorsDims.priors.c});

            SetInputTensors(inputTensors);
            SetOutputTensors(outputTensors);

            std::map<std::string, std::string> layerParams = {
                {"flatten",  std::to_string(opParams.flatten)},
                {"h",        std::to_string(opParams.grid_h)},
                {"w",        std::to_string(opParams.grid_w)},
                {"stride_y", std::to_string(opParams.stride_h)},
                {"stride_x", std::to_string(opParams.stride_w)}
            };

            ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ExperimentalDetectronPriorGridGenerator").params(layerParams)));

            /* Input data generating */
            for (auto blob : _inputMap) {
                if (blob.second == _inputMap.begin()->second) {
                    int w = inputTensorsDims.featureMap.w;
                    int h = inputTensorsDims.featureMap.h;

                    if (strcmp(InputsAvailable, MISSING_FEATURE_MAP) == 0) {
                        w = opParams.grid_w;
                        h = opParams.grid_h;
                    }

                    genPriors(blob.second, w, h, numPriors);
                } else {
                    GenRandomData(blob.second);
                }
            }

            std::vector<InferenceEngine::Blob::Ptr> refInputBlobs;
            std::vector<InferenceEngine::Blob::Ptr> refOutputBlobs;

            for (auto blob : _inputMap) {
                auto _refInputBlob = make_shared_blob<ie_fp16>({Precision::FP16,
                                                                blob.second->getTensorDesc().getDims(),
                                                                blob.second->getTensorDesc().getLayout()},
                                                                blob.second->buffer());
                refInputBlobs.push_back(_refInputBlob);
            }

            for (auto blob : _outputMap) {
                auto refOutputBlob = make_shared_blob<ie_fp16>({Precision::FP16,
                                                                blob.second->getTensorDesc().getDims(),
                                                                blob.second->getTensorDesc().getLayout()});
                refOutputBlob->allocate();
                refOutputBlobs.push_back(refOutputBlob);
            }

            ref_ExpPriorGridGenerator(refInputBlobs,
                                      refOutputBlobs,
                                      opParams.grid_w,
                                      opParams.grid_h,
                                      opParams.stride_w,
                                      opParams.stride_h);

            ASSERT_TRUE(Infer());
            CompareCommonAbsolute(_outputMap.begin()->second, refOutputBlobs[0], ERROR_BOUND);
        }
};

class myriadLayersTestsExpPriorGridGeneratorAllInputs_smoke : public myriadLayersTestsExpPriorGridGen<ALL_INPUTS>
{
};

class myriadLayersTestsExpPriorGridGeneratorNoFeatureMap_smoke : public myriadLayersTestsExpPriorGridGen<MISSING_FEATURE_MAP>
{
};

class myriadLayersTestsExpPriorGridGeneratorNoInputImage_smoke : public myriadLayersTestsExpPriorGridGen<MISSING_INPUT_IMAGE>
{
};


TEST_P(myriadLayersTestsExpPriorGridGeneratorAllInputs_smoke, AllThreeInputs) {
    runTest();
}

TEST_P(myriadLayersTestsExpPriorGridGeneratorNoFeatureMap_smoke, MissingFeatureMap) {
    runTest();
}

TEST_P(myriadLayersTestsExpPriorGridGeneratorNoInputImage_smoke, MissingImageInput) {
    runTest();
}

static std::vector<InputDims> s_ExpPriorGridGeneratorLayerInputs = {
    {
        InputDims(
            Dims({3, 4}),          // priors
            Dims({1, 128, 8, 8}),  // feature map
            Dims({1, 3, 480, 480}) // im_data
        )
    },
    {
        InputDims(
            Dims({3, 4}),           // priors
            Dims({1, 128, 60, 60}), // feature map
            Dims({1, 3, 480, 480})  // im_data
        )
    },
    {
        InputDims(
            Dims({3, 4}),             // priors
            Dims({1, 128, 120, 120}), // feature map
            Dims({1, 3, 480, 480})    // im_data
        )
    },
    {
        InputDims(
            Dims({64, 4}),          // priors
            Dims({1, 128, 16, 16}), // feature map
            Dims({1, 3, 480, 480})  // im_data
        )
    },
};

static std::vector<PriorGridGeneratorParam> s_ExpPriorGridGeneratorLayerParam = {
    {1, 0, 0, 16.0f, 16.0f}, {1, 0, 0, 8.0f, 8.0f}, {1, 0, 0, 4.0f, 4.0f},
    {1, 8, 8, 64.0f, 64.0f}, {1, 10, 16, 0.0f, 0.0f}
};

static std::vector<InputDims> s_ExpPriorGridGenLayerNoFMInputs = {
    {
        InputDims(
            Dims({3, 4}),           // priors
            Dims({1, 1, 1, 1}),     // feature map
            Dims({1, 3, 512, 512})  // im_data
        )
    }
};

static std::vector<PriorGridGeneratorParam> s_ExpPriorGridGenLayerNoFMParam = {
    {1, 16, 16, .0f, .0f}, {1, 32, 32, 4.0f, 4.0f}
};

static std::vector<InputDims> s_ExpPriorGridGenLayerNoInputImage = {
    {
        InputDims(
            Dims({3, 4}),           // priors
            Dims({1, 128, 60, 60}), // feature map
            Dims({1, 1, 1, 1})      // im_data
        )
    },
    {
        InputDims(
            Dims({64, 4}),          // priors
            Dims({1, 128, 16, 16}), // feature map
            Dims({1, 3, 480, 480})  // im_data
        )
    },
};

static std::vector<PriorGridGeneratorParam> s_ExpPriorGridGenLayerNoInputImageParam = {
    {1, 0, 0, 8.0f, 8.0f}, {1, 32, 32, 4.0f, 4.0f}
};
