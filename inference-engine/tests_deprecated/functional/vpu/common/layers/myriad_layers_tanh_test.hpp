// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#define BOUND (10.0f)
#define ERROR_BOUND (1.2e-3f)
#define ERROR_BOUND_WITH_TANH (1.0e-3f)
using namespace InferenceEngine;

class myriadLayersTestsTanh_smoke: public myriadLayersTests_nightly,
                             public testing::WithParamInterface<SizeVector> {
};

TEST_P(myriadLayersTestsTanh_smoke, TestsTanh)
{
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    auto p = ::testing::WithParamInterface<SizeVector>::GetParam();
    SetInputTensors({p});
    SetOutputTensors({p});

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("TanH")));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());
    /* output check */
    ref_tanh(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_tanhParams = {
    {{4, 1, 16, 16}},
    {{4, 2, 16, 16}},
    {{4, 3, 16, 16}},
    {{4, 4, 1, 53, 16}},
    {{4, 4, 2, 53, 16}},
    {{4, 4, 3, 53, 16}},
    {{4, 4, 1, 224, 224}},
    {{4, 4, 4, 2, 224, 224}},
    {{4, 4, 4, 3, 224, 224}},
    {{4, 4, 4, 1, 224, 235}},
    {{4, 4, 4, 2, 224, 235}},
    {{4, 4, 4, 3, 224, 235}},
    {{1, 1, 277, 230}},
    {{1, 2, 277, 230}},
    {{1, 3, 277, 230}}

};

static std::vector<InferenceEngine::SizeVector> s_convolutionTensors = {
    {{1, 8, 4, 16}, {16, 8, 16}}  //NCHW
};

/* tests subset to check 2 layers operation invocation */
/* additional tests for 2D and 3D tensors added        */
static std::vector<int32_t> s_dimensionsFC = {
    4, 3
};

static std::vector<int32_t> s_addBiasFC = {
    1, 0
};

/* to decrease tests duration and tests amount */
static std::vector<fcon_test_params> s_fcTestParamsSubset = {
    {{1, 1, 16, 8},     8, 0.02f},
    {{1, 1, 8, 40},     8, 0.02f},
    {{1, 4, 8, 16},     4, 0.065f},
    {{1, 16, 16, 16},  16, 0.36f},
    {{1, 16, 8, 8},    8, 0.065f}
};

class myriadLayerConvolutionWithTanH_smoke: public ConvolutionTest<IRVersion>{
};

TEST_P(myriadLayerConvolutionWithTanH_smoke, Convolution) {
    auto param = GetParam();
    _irVersion = std::get<6>(param);

    _testNet.addLayer(LayerInitParams("TanH")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_tanh_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

class myriadLayersTestsMaxPoolingWithTanh_smoke: public PoolingTest<POOLING_MAX>{
};

class myriadLayersTestsAvgPoolingWithTanh_smoke: public PoolingTest<POOLING_AVG>{
};

TEST_P(myriadLayersTestsMaxPoolingWithTanh_smoke, TestsMaxPoolingWithTanh)
{
    _testNet.addLayer(LayerInitParams("TanH")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_tanh_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_TANH);
}

TEST_P(myriadLayersTestsAvgPoolingWithTanh_smoke, TestsAvgPoolingWithTanh)
{
    _testNet.addLayer(LayerInitParams("TanH")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_tanh_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_TANH);
}

class myriadLayerFullyConnectedWithTanH_smoke: public FCTest<>{
};

TEST_P(myriadLayerFullyConnectedWithTanH_smoke, TestsFullyConnected)
{
    _testNet.addLayer(LayerInitParams("TanH")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_tanh_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), _par.error_bound);
}
