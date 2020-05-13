// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <algorithm>

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(ChannelSharedPrelu, int);
typedef myriadLayerTestBaseWithParam<tuple<SizeVector, ChannelSharedPrelu >> myriadLayerPReLU_smoke;

TEST_P(myriadLayerPReLU_smoke, PReLU) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    SizeVector dims = get<0>(GetParam());
    int channel_shared = get<1>(GetParam());

    SetInputTensors({dims});
    SetOutputTensors({dims});

    int num_weights = channel_shared ? 1 : dims[dims.size() - 3];
    
    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(GenWeights(num_weights));
    uint16_t* weights = weights_ptr->data().as<uint16_t*>();

    std::map<std::string, std::string> layer_params = {{"channel_shared", std::to_string(channel_shared)}};
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("PReLU")
                                        .params(layer_params)
                                        .weights(weights_ptr->byteSize() /sizeof (uint16_t)),
                                        {},
                                        weights_ptr));
    SetFirstInputToRange(0, 5.0f);
    ASSERT_TRUE(Infer());

    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;

    ref_PReLU(inputBlob, _refBlob, weights, num_weights);
    CompareCommonAbsolute(outputBlob, _refBlob, 0);
}

static std::vector<InferenceEngine::SizeVector> s_PReLUTensors = {
    {
        {13, 38, 38},
        {1, 13, 77,  99},
        {4,  3, 11,   8},
        {3,  11, 11,  8, 8}
    },
};

struct  PReLULayerDef {
    ParamsStruct list;
}PReLULayer;

static std::vector<PReLULayerDef> s_PReluLayerParams = {
    {{{PRELU_PARAM, "0"}}},
    {{{PRELU_PARAM, "1"}}}
};

class myriadLayerFullyConnectedWithPReLU_smoke: public FCTest<PReLULayerDef>{
};

#define TEST_BODY \
    int channel_shared = 0;\
    if (!extraLayerParams.list.empty()) {\
        auto iter = extraLayerParams.list.find(PRELU_PARAM);\
        if (iter != extraLayerParams.list.end()) {\
             channel_shared = std::stoi(iter->second);\
        }\
    }\
    size_t weightsSize = 1;\
    if (channel_shared == 0) {\
        int32_t OW;\
        int32_t OH;\
        int32_t OC;\
        get_dims(_output_tensor, OW, OH, OC);\
        weightsSize = OC;\
    }\
    _testNet.addLayer(LayerInitParams("PReLU")\
             .params(extraLayerParams.list)\
             .weights(weightsSize).fillWeights(defaultWeightsRange)\
             .in({_output_tensor})\
             .out({_output_tensor}),\
             ref_PReLU_wrap);\
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));

TEST_P(myriadLayerFullyConnectedWithPReLU_smoke, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), _par.error_bound);
}

#define ERROR_BOUND_WITH_RELU (4.e-3f)

class myriadLayersTestsMaxPoolingWithPReLU_smoke: public PoolingTest<POOLING_MAX, PReLULayerDef>{
};

class myriadLayersTestsAvgPoolingWithPReLU_smoke: public PoolingTest<POOLING_AVG, PReLULayerDef>{
};

TEST_P(myriadLayersTestsMaxPoolingWithPReLU_smoke, TestsMaxPoolingWithPReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, vpu::LayoutPreference, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

TEST_P(myriadLayersTestsAvgPoolingWithPReLU_smoke, TestsAvgPoolingWithPReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, vpu::LayoutPreference, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    TEST_BODY;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

class myriadLayerConvolutionWithPReLU_smoke: public ConvolutionTest<PReLULayerDef>{
};

TEST_P(myriadLayerConvolutionWithPReLU_smoke, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, PReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<6>(p);
    TEST_BODY;
    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}
