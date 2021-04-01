// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

struct clamp_test_params {
    float min;
    float max;
    friend std::ostream& operator<<(std::ostream& os, clamp_test_params const& tst)
    {
        return os << " min=" << tst.min
                  << ", max=" << tst.max;
    };
};

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, clamp_test_params>> myriadLayersTestsClampParams_smoke;

TEST_P(myriadLayersTestsClampParams_smoke, TestsClamp) {
    _config[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
    auto param = GetParam();
    SizeVector tensor = std::get<0>(param);
    clamp_test_params p = std::get<1>(param);

    std::map<std::string, std::string> params;
    params["min"] = std::to_string(p.min);
    params["max"] = std::to_string(p.max);

    SetInputTensors({tensor});
    SetOutputTensors({tensor});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Clamp").params(params)));
    /* input data preparation */
    SetFirstInputToRange(-100.f, 100.f);
    ASSERT_TRUE(Infer());

    /* output check */
    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    auto inputBlob  = _inputMap[_inputsInfo.begin()->first];

    ref_Clamp(inputBlob, _refBlob, p.min, p.max);

    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_clampTensors = {
    {{1, 3, 10, 15}},
    {{5, 6, 2, 3, 10, 15}},
};

static std::vector<clamp_test_params> s_clampParams = {
    {0.f, 6.0f},
    {-10.f, 17.0f}
};
