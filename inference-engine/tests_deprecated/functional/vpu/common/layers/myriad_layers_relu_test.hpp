// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

const std::string relu_param = "negative_slope";

class myriadLayersTestsReLUMergeWithBias_smoke : public myriadLayersTests_nightly {
public:
    void RunTest(const std::string& model, size_t num_weights, size_t num_bias) {
        TBlob<uint8_t>::Ptr weights(GenWeights(num_weights + num_bias));

        ASSERT_NO_THROW(readNetwork(model, weights));

        const auto& network = _cnnNetwork;

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["input"]->setPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["relu"]->setPrecision(Precision::FP16);

        ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network,
                { {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)},
                  {InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(NO)} }));

        ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());
        
        ASSERT_NO_THROW(_inferRequest.Infer());
        
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(perfMap = _inferRequest.GetPerformanceCounts());
        
        {
            auto reluAndBiasLayerIt = perfMap.find("relu+Bias");
            ASSERT_TRUE(reluAndBiasLayerIt != perfMap.end());
            EXPECT_EQ(InferenceEngineProfileInfo::EXECUTED, reluAndBiasLayerIt->second.status);
        }
    }
};

#define ERROR_BOUND (1.e-4f)

using namespace InferenceEngine;

struct ReLULayerDef {
    ParamsStruct list;
}ReLULayer;

static std::vector<ReLULayerDef> s_reluLayerParams = {
    {{{"negative_slope", "0.0"}}},
    {{{"negative_slope", "0.1"}}},
};

typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, ReLULayerDef>> myriadLayerReLU_smoke;

TEST_P(myriadLayerReLU_smoke, ReLU) {
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    auto input_dims = std::get<0>(GetParam());
    auto extraLayerParams = std::get<1>(GetParam());
    IN_OUT_desc input_tensor;
    input_tensor.push_back(input_dims);

    /* Copy is implemented to perform filling of the output buffer */
    _testNet.addLayer(LayerInitParams("Copy")
             .in(input_tensor)
             .out(input_tensor),
             ref_copy_wrap);

    _testNet.addLayer(LayerInitParams("ReLU")
             .params(extraLayerParams.list)
             .in({input_tensor})
             .out({input_tensor}),
             ref_ReLU_wrap);

    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND);
}

static std::vector<InferenceEngine::SizeVector> s_copyTensors = {
    {
        {16, 18},
        {1, 8, 16, 32},
        {12, 32, 64, 32, 12},
        {24, 32, 16},
    },
};

class myriadLayerFullyConnectedWithReLU_smoke: public FCTest<ReLULayerDef>{
};

TEST_P(myriadLayerFullyConnectedWithReLU_smoke, TestsFullyConnected)
{
    auto p = ::testing::WithParamInterface<std::tuple<fcon_test_params, int32_t, int32_t, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    _testNet.addLayer(LayerInitParams("ReLU")
             .params(extraLayerParams.list)
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_ReLU_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), _par.error_bound);
}

#define ERROR_BOUND_WITH_RELU (4.e-3f)

class myriadLayersTestsMaxPoolingWithReLU_smoke: public PoolingTest<POOLING_MAX, ReLULayerDef>{
};

class myriadLayersTestsAvgPoolingWithReLU_smoke: public PoolingTest<POOLING_AVG, ReLULayerDef>{
};

TEST_P(myriadLayersTestsMaxPoolingWithReLU_smoke, TestsMaxPoolingWithReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, vpu::LayoutPreference, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    _testNet.addLayer(LayerInitParams("ReLU")
             .params(extraLayerParams.list)
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_ReLU_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

TEST_P(myriadLayersTestsAvgPoolingWithReLU_smoke, TestsAvgPoolingWithReLU)
{
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, pooling_layer_params, vpu::LayoutPreference, ReLULayerDef>>::GetParam();
    auto extraLayerParams = std::get<3>(p);
    _testNet.addLayer(LayerInitParams("ReLU")
             .params(extraLayerParams.list)
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_ReLU_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_RELU);
}

class myriadLayerConvolutionWithReLU_smoke: public ConvolutionTest<ReLULayerDef>{
};

TEST_P(myriadLayerConvolutionWithReLU_smoke, Convolution) {
    auto p = ::testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, param_size, param_size, param_size, uint32_t, uint32_t, ReLULayerDef>>::GetParam();
    auto ReLUParam = std::get<6>(p);
    _testNet.addLayer(LayerInitParams("ReLU")
             .params(ReLUParam.list)
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_ReLU_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}
