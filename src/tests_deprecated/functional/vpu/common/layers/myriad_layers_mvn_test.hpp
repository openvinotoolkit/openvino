// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0.2f

static void refMVN(const Blob::Ptr src,
                   Blob::Ptr dst,
                   int across_channels, int normalize_variance, const float eps, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    int32_t IB = 1;
    get_dims(src, IW, IH, IC);

    float* mean_buf = new float[IW*IH*IC];

    for (int b = 0; b < IB; b++)
    {
        // Calculate mean value
        if (across_channels)
        {
            float mean = 0;
            for (int c = 0; c < IC; c++) {
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean += s;
                    }
                }
            }
            mean /= IC*IH*IW;
            for (int c = 0; c < IC; c++) {
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean_buf[ind] = s - mean;
                        dst_data[ind] = PrecisionUtils::f32tof16(s - mean);
                    }
                }
            }
        }
        else {
            for (int c = 0; c < IC; c++)
            {
                float mean = 0;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean += s;
                    }
                }
                mean /= IH*IW;
                for (int h = 0; h < IH; h++) {
                    for (int w = 0; w < IW; w++) {
                        int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                        float s = PrecisionUtils::f16tof32(src_data[ind]);
                        mean_buf[ind] = s - mean;
                        dst_data[ind] = PrecisionUtils::f32tof16(s - mean);
                    }
                }
            }
        }
    }

    if (normalize_variance)
    {
        for (int b = 0; b < IB; b++)
        {
            // Calculate variances value
            if (across_channels)
            {
                float variance = 0;
                for (int c = 0; c < IC; c++) {
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            variance += mean_buf[ind] * mean_buf[ind];
                        }
                    }
                }
                variance /= IC*IH*IW;
                variance = sqrtf(variance);//std::pow(variance, 0.5f);
                variance += eps;
                for (int c = 0; c < IC; c++) {
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            dst_data[ind] = PrecisionUtils::f32tof16(mean_buf[ind] / variance);
                        }
                    }
                }
            }
            else {
                for (int c = 0; c < IC; c++)
                {
                    float variance = 0;
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            variance += mean_buf[ind] * mean_buf[ind];
                        }
                    }
                    variance /= IH*IW;
                    variance = sqrtf(variance);
                    variance += eps;
                    for (int h = 0; h < IH; h++) {
                        for (int w = 0; w < IW; w++) {
                            int ind = isCHW ? c*IH*IW + h*IW + w : h*IW*IC + w*IC + c;
                            dst_data[ind] = PrecisionUtils::f32tof16(mean_buf[ind] / variance);
                        }
                    }
                }
            }
        }
    }

    delete[] mean_buf;
}

PRETTY_PARAM(AcrossChannels, int)
PRETTY_PARAM(Normalize, int)
PRETTY_PARAM(Epsilon, float)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, AcrossChannels, Normalize, Epsilon, IRVersion, std::string>> myriadLayersTestsMVN_smoke;

TEST_P(myriadLayersTestsMVN_smoke, DISABLED_MVN)
{
    tensor_test_params dims  = std::get<0>(GetParam());
    int acrossChannels       = std::get<1>(GetParam());
    int normalize            = std::get<2>(GetParam());
    float eps                = std::get<3>(GetParam());
    _irVersion               = std::get<4>(GetParam());
    std::string customConfig = std::get<5>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["across_channels"] = std::to_string(acrossChannels);
    params["normalize_variance"] = std::to_string(normalize);
    params["eps"] = std::to_string(eps);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("MVN").params(params)));
    ASSERT_NO_FATAL_FAILURE(SetFirstInputToRange(0, 256));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refMVN(_inputMap.begin()->second, _refBlob, acrossChannels, normalize, eps, false));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_MVNTensors = {
        {{1, 3, 512, 896}}
};

static std::vector<AcrossChannels> s_MVN_acrossChannels = { 0, 1};
static std::vector<Normalize> s_MVN_normalize = { 0, 1};
static std::vector<Epsilon> s_MVN_epsilon = { 1.0e-10, 1.0e-8, 1.0e-7, 1.0e-5, 1.0e-3};
static std::vector<std::string> s_MVNCustomConfig = {
    "",
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};
