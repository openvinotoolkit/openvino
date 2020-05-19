// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 1e-3

PRETTY_PARAM(Factor, float)
PRETTY_PARAM(Antialias, int)
PRETTY_PARAM(HwOptimization, bool);
PRETTY_PARAM(CustomConfig, std::string);

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Factor, Antialias, HwOptimization, CustomConfig>>
	myriadResampleLayerTests_smoke;

static inline float triangleCoeff(float x)
{
    return (1.0f - fabsf(x));
}

void refResample(const Blob::Ptr src, Blob::Ptr dst, int antialias) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    const auto& src_dims = src->getTensorDesc().getDims();
    const auto& dst_dims = dst->getTensorDesc().getDims();
    int OH = dst_dims[2];
    int OW = dst_dims[3];

    int C  = src_dims[1];
    int IH = src_dims[2];
    int IW = src_dims[3];

    if (IH == OH && IW == OW)
    {
    	std::copy(src_data, src_data + C*IH*IW, output_sequences);
        return;
    }

    const float fy = static_cast<float>(IH) / static_cast<float>(OH);
    const float fx = static_cast<float>(IW) / static_cast<float>(OW);

    float ax = 1.0f / fx;
    float ay = 1.0f / fy;

    int rx = (fx < 1.0f) ? 2 : ceil((1.0f)/ax);
    int ry = (fy < 1.0f) ? 2 : ceil((1.0f)/ay);

    for (int c = 0; c < C; c++)
    {
        const ie_fp16* in_ptr = src_data + IW*IH*c;
        ie_fp16* out_ptr = output_sequences + OW*OH*c;

        for (int oy = 0; oy < OH; oy++)
        {
            for (int ox = 0; ox < OW; ox++)
            {
                float ix = ox*fx + fx / 2.0f - 0.5f;
                float iy = oy*fy + fy / 2.0f - 0.5f;

                int ix_r = (int)(round(ix));
                int iy_r = (int)(round(iy));

                float sum=0;
                float wsum=0;

                if(antialias){
                    for (int y = iy_r - ry; y <= iy_r + ry; y++)
                    {
                        for (int x = ix_r - rx; x <= ix_r + rx; x++)
                        {
                            if (y < 0 || x < 0) continue;
                            if (y >= (int)IH || x >= (int)IW) continue;

                            float dx = ix - x;
                            float dy = iy - y;

                            float w = ax*triangleCoeff(ax*dx) * ay*triangleCoeff(ay*dy);

                            sum += w * PrecisionUtils::f16tof32(in_ptr[y*IW + x]);
                            wsum += w;
                        }
                    }
                    out_ptr[oy * OW + ox] = PrecisionUtils::f32tof16((!wsum) ? 0.0f : (sum / wsum));
                }
                else{
                    out_ptr[oy * OW + ox] = in_ptr[iy_r * IW + ix_r];
                }
            }
        }
    }
}

TEST_P(myriadResampleLayerTests_smoke, Resample) {
    const SizeVector inputDims = std::get<0>(GetParam());
    const float factor = std::get<1>(GetParam());
    const bool antialias = std::get<2>(GetParam());
    const bool hwOptimization = std::get<3>(GetParam());
    const std::string customConfig = std::get<4>(GetParam());

    ASSERT_GT(factor, 0);

    if (customConfig.empty() && antialias) {
        GTEST_SKIP() << "Native Resample with antialiasing is not supported";
    }

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }

    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    const auto outputDims = SizeVector{inputDims[0],
                                       inputDims[1],
                                       (size_t)(inputDims[2] * factor),
                                       (size_t)(inputDims[3] * factor)};

    SetInputTensors({inputDims});
    SetOutputTensors({outputDims});

    std::map<std::string, std::string> params;
    params["antialias"] = std::to_string((int)antialias);
    params["factor"] = std::to_string(factor);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Resample").params(params),
                                                   NetworkInitParams()
                                                        .useHWOpt(hwOptimization)
                                                        .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refResample(_inputMap.begin()->second, _refBlob, antialias));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<SizeVector> s_ResampleInput = {
        {1, 128, 26, 26},
        {1, 64, 52, 52},
        {1, 23, 14, 14}
};

static std::vector<CustomConfig> s_CustomConfig = {
    {""},
#ifdef VPU_HAS_CUSTOM_KERNELS
   getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

