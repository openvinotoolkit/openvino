// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

#define BOUND (5.0f)
#define ERROR_BOUND (1.0f)

using namespace InferenceEngine;

struct pwr_test_params {
    float power;
    float scale;
    float shift;
    friend std::ostream& operator<<(std::ostream& os, pwr_test_params const& tst)
    {
        return os << " power=" << tst.power
                  << ", scale=" << tst.scale
                  << ", shift=" << tst.shift;
    };
};

static void gen_ref_power(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          pwr_test_params& p) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->getTensorDesc().getDims().size(), dst->getTensorDesc().getDims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    //Compute y = (shift + scale * x) ^ power
    for (size_t indx = 0; indx < src->size(); indx++) {
        dstData[indx] = PrecisionUtils::f32tof16(pow((p.shift + p.scale * PrecisionUtils::f16tof32(srcData[indx])), p.power));
    }
}

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, pwr_test_params>> myriadLayersTestsPowerParams_nightly;

TEST_P(myriadLayersTestsPowerParams_nightly, TestsPower) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    auto param = GetParam();
    SizeVector tensor = std::get<0>(param);
    pwr_test_params p = std::get<1>(param);

    std::map<std::string, std::string> params;
    params["power"] = std::to_string(p.power);
    params["scale"] = std::to_string(p.scale);
    params["shift"] = std::to_string(p.shift);

    SetInputTensors({tensor});
    SetOutputTensors({tensor});
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Power").params(params)));
    /* input data preparation */
    SetFirstInputToRange(0, BOUND);
    ASSERT_TRUE(Infer());

    /* output check */
    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    auto inputBlob = _inputMap[_inputsInfo.begin()->first];

    gen_ref_power(inputBlob, _refBlob, p);

    float eps_err = ERROR_BOUND;

    /* for "dst = -src" case results have to be equal */
    if ((p.power == 1.0f) && ((p.scale == -1.0f) || (p.scale == 1.0f)) && (p.shift == 0.0f))
        eps_err = 0.0f;

    CompareCommonAbsolute(outputBlob, _refBlob, eps_err);
}

static std::vector<SizeVector> s_powerTensors = {
    {{1, 1, 32*10, 16*10}},
    {{6, 5, 4, 3, 32, 16}},
};

static std::vector<pwr_test_params> s_powerParams = {
    {0.f,  1.0f,  0.0f},
    {1.f,  1.0f,  0.0f},
    {1.f, -1.0f,  0.0f},
    {1.f, -1.0f, 0.71f},
    {2.f, -1.4f,  3.1f},
    {3.f,  1.1f, -2.1f},
    {7.f,  0.1f,  1.0f},
    {-8.f,  0.1f,  1.0f},
    /* various power */
    { 3.1f,  0.5f, 3.0f},
    { 0.50f, 0.50f, 1.0f},
    {-1.50f, 1.50f, 1.0f},
    { 10.50f,  0.1f, 0.1f}
};
