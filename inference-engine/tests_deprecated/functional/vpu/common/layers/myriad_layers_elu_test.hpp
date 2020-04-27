// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

#define BOUND (5.0f)
#define ERROR_BOUND (8e-3)

using namespace InferenceEngine;

PRETTY_PARAM(alpha, float);

void gen_ref_elu(const InferenceEngine::Blob::Ptr src,
                        InferenceEngine::Blob::Ptr dst,
                        alpha p) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    const int16_t *srcData = src->buffer();
    int16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    for (size_t indx = 0; indx < src->size(); indx++) {
        float src_val = PrecisionUtils::f16tof32(srcData[indx]);
        dstData[indx] = PrecisionUtils::f32tof16(src_val > 0 ? src_val : p * (expf(src_val) - 1.f));
    }
}

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, alpha>> myriadLayersTestsELUParams;

TEST_P(myriadLayersTestsELUParams, TestsELU) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);

    auto param = GetParam();
    SizeVector tensor = std::get<0>(param);
    alpha p = std::get<1>(param);

    std::map<std::string, std::string> params;
    params["alpha"] = std::to_string(p);

    SetInputTensors({tensor});
    SetOutputTensors({tensor});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ELU").params(params)));
    /* input data preparation */
    SetFirstInputToRange(-BOUND, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];
    
    gen_ref_elu(inputBlob, _refBlob, p);
    CompareCommonAbsolute(outputBlob, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_powerTensors = {
    {{6, 5, 4, 3, 40, 43}},
    {{6, 5, 4, 3}},
    {{6, 5, 4}},
};

static std::vector<alpha> s_powerParams = {
    0.1f,
    0.0f,
    1.0f,
    5.0f,
};
