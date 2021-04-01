// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <cmath>
#include <algorithm>

#define BOUND (10.0f)
#define ERROR_BOUND (1.e-3f)
#define ERROR_BOUND_WITH_SIGMOID (1.e-3f)

using namespace InferenceEngine;

class myriadLayersTestsSigmoid_smoke: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<InferenceEngine::SizeVector> {
public:
};

TEST_P(myriadLayersTestsSigmoid_smoke, TestsSigmoid)
{
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

    SizeVector p = GetParam();
    SetInputTensors({p});
    SetOutputTensors({p});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Sigmoid")));
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    ref_sigmoid(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<InferenceEngine::SizeVector> s_sigmoidParams = {
    {{3, 1, 16, 16}},
    {{3, 2, 16, 16}},
    {{3, 3, 16, 16}},
    {{3, 1, 53, 16}},
    {{3, 2, 53, 16}},
    {{3, 3, 53, 16}},
    {{4, 4, 1, 224, 224}},
    {{4, 4, 2, 224, 224}},
    {{4, 4, 3, 224, 224}},
    {{1, 224, 235}},
    {{2, 224, 235}},
    {{3, 224, 235}},
    {{1, 1, 277, 230}},
    {{1, 2, 277, 230}},
    {{1, 3, 277, 230}}
};

class myriadLayersTestsMaxPoolingWithSigmoid_smoke: public PoolingTest<POOLING_MAX>{
};

class myriadLayersTestsAvgPoolingWithSigmoid_smoke: public PoolingTest<POOLING_AVG>{
};

TEST_P(myriadLayersTestsMaxPoolingWithSigmoid_smoke, TestsMaxPoolingWithSigmoid)
{
    _testNet.addLayer(LayerInitParams("Sigmoid")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_sigmoid_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_SIGMOID);
}

TEST_P(myriadLayersTestsAvgPoolingWithSigmoid_smoke, TestsAvgPoolingWithSigmoid)
{
    _testNet.addLayer(LayerInitParams("Sigmoid")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_sigmoid_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), ERROR_BOUND_WITH_SIGMOID);
}

class myriadLayerConvolutionWithSigmoid_smoke: public ConvolutionTest<IRVersion>{
};

TEST_P(myriadLayerConvolutionWithSigmoid_smoke, Convolution) {
    _irVersion = std::get<6>(GetParam());
    _testNet.addLayer(LayerInitParams("Sigmoid")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_sigmoid_wrap);

    float maxerr = 0;
    if (group == 1)
        maxerr = 0.00055 * IC * kernel.x * kernel.y;
    else // TODO: currently dephConv is slightly less accurate
        maxerr = 0.00066 * (IC / group) * kernel.x * kernel.y;
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), maxerr);
}

class myriadLayerFullyConnectedWithSigmoid_smoke: public FCTest<>{
};

TEST_P(myriadLayerFullyConnectedWithSigmoid_smoke, TestsFullyConnected)
{
    _testNet.addLayer(LayerInitParams("Sigmoid")
             .in({_output_tensor})
             .out({_output_tensor}),
             ref_sigmoid_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams()));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), _par.error_bound);
}
