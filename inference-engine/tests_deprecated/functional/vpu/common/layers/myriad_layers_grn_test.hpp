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
PRETTY_PARAM(CustomConfig, std::string)

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Bias, IRVersion, CustomConfig>> myriadLayersTestsGRN_smoke;

TEST_P(myriadLayersTestsGRN_smoke, GRN) {
    const SizeVector dims = std::get<0>(GetParam());
	const float bias = std::get<1>(GetParam());
	_irVersion = std::get<2>(GetParam());
	const std::string customConfig = std::get<3>(GetParam());

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensors({dims});
    SetOutputTensors({dims});

    std::map<std::string, std::string> params;
    params["bias"] = std::to_string(bias);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("GRN").params(params),
												   NetworkInitParams()
												   .layoutPreference(vpu::LayoutPreference::ChannelMajor)
												   .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refGRN(_inputMap.begin()->second, _refBlob, bias, true));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<CustomConfig> s_CustomConfig = {
	{""} ,
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

static std::vector<SizeVector> s_GRNInputs = {
        {1, 3, 16, 224},
        {1, 24, 128, 224},
};