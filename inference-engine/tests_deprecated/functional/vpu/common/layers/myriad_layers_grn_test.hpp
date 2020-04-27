// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3f

static void refGRN(const Blob::Ptr src,
                         Blob::Ptr dst,
                   float bias, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    for (uint32_t h = 0; h < IH; h++) {
        for (uint32_t w = 0; w < IW; w++) {
            float variance = 1e-9f;
            for (uint32_t c = 0; c < IC; c++) {
                int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                float s = PrecisionUtils::f16tof32(src_data[ind]);
                variance += powf(s, 2);
            }
            variance = sqrtf(variance + bias);
            for (uint32_t c = 0; c < IC; c++) {
                int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;

                float s = PrecisionUtils::f16tof32(src_data[ind]);
                float result = s / variance;

                dst_data[ind] = PrecisionUtils::f32tof16(result);
            }
        }
    }
}

PRETTY_PARAM(Bias, float)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Bias, std::string>> myriadLayersTestsGRN_nightly;

TEST_P(myriadLayersTestsGRN_nightly, GRN) {
    tensor_test_params dims  = std::get<0>(GetParam());
    float bias               = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["bias"] = std::to_string(bias);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("GRN").params(params)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refGRN(_inputMap.begin()->second, _refBlob, bias, false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_GRNTensors = {
        {{1, 3, 16, 224}},
        {{1, 24, 128, 224}},
};

static std::vector<Bias> s_GRN_bias = {
        0.5f, 10.f
};

static std::vector<std::string> s_MVNCustomConfig = {
    "" ,
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};
