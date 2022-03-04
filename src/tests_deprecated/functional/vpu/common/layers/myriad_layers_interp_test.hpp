// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(layoutPreference, vpu::LayoutPreference)
PRETTY_PARAM(SizeInputOutput, interp_test_params)
PRETTY_PARAM(align_corners, bool)

typedef myriadLayerTestBaseWithParam<tuple<interp_test_params, layoutPreference, align_corners>> myriadLayersTestsInterp_smoke;


void ref_interp(const Blob::Ptr src,
                Blob::Ptr dst, bool align_corners) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *dst_data = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t OW = 0;
    int32_t OH = 0;
    int32_t OC = 0;
    int32_t N = 1;

    get_dims(src, IW, IH, IC);
    get_dims(dst, OW, OH, OC);
    int32_t C = IC;

    if (IH == OH && IW == OW)
    {
        for (size_t b = 0; b < N; b++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < IH; h++) {
                    for (size_t w = 0; w < IW; w++) {
                        size_t oidx = c + w * C + h * C * OW;
                        size_t iidx = c + w * C + h * C * IW;
                        ASSERT_LT(iidx, src->size());
                        ASSERT_LT(oidx, dst->size());
                        dst_data[oidx] = src_data[iidx];
                    }
                }
            }
        }
        return;
    }

    const float rh = (OH > 1 && align_corners) ? static_cast<float>(IH - 1) / (OH - 1) : static_cast<float>(IH) / OH;
    const float rw = (OW > 1 && align_corners) ? static_cast<float>(IW - 1) / (OW - 1) : static_cast<float>(IW) / OW;

    for (size_t b = 0; b < N; ++b) {
        for (size_t h = 0; h < OH; h++) {
            float fh = rh * h;
            size_t ih0 = static_cast<size_t>(fh);
            size_t ih1 = (ih0 < IH - 1) ? ih0 + 1 : ih0;

            float h_lambda0 = fh - ih0;
            float h_lambda1 = 1.0f - h_lambda0;

            for (size_t w = 0; w < OW; w++) {
                float fw = rw * w;
                size_t iw0 = static_cast<size_t>(fw);
                size_t iw1 = (iw0 < IW - 1) ? iw0 + 1 : iw0;

                float w_lambda0 = fw - iw0;
                float w_lambda1 = 1.0f - w_lambda0;

                for (size_t c = 0; c < C; c++) {
                    size_t iidx00 = c + iw0 * C + ih0 * C * IW;
                    size_t iidx01 = c + iw1 * C + ih0 * C * IW;
                    size_t iidx10 = c + iw0 * C + ih1 * C * IW;
                    size_t iidx11 = c + iw1 * C + ih1 * C * IW;
                    ASSERT_LT(iidx00, src->size());
                    ASSERT_LT(iidx01, src->size());
                    ASSERT_LT(iidx10, src->size());
                    ASSERT_LT(iidx11, src->size());

                    float src00 = PrecisionUtils::f16tof32(src_data[iidx00]);
                    float src01 = PrecisionUtils::f16tof32(src_data[iidx01]);
                    float src10 = PrecisionUtils::f16tof32(src_data[iidx10]);
                    float src11 = PrecisionUtils::f16tof32(src_data[iidx11]);

                    size_t oidx = c + w * C + h * C * OW;
                    ASSERT_LT(oidx, dst->size());

                    dst_data[oidx] = PrecisionUtils::f32tof16(h_lambda1 * (w_lambda1 * src00 + w_lambda0 * src01) +
                                                              h_lambda0 * (w_lambda1 * src10 + w_lambda0 * src11));
                }
            }
        }
    }
}

TEST_P(myriadLayersTestsInterp_smoke, Interp)
{
    interp_test_params test_params = get<0>(GetParam());
    auto layoutPreference = get<1>(GetParam());
    bool align_corner = get<2>(GetParam());

    std::map<std::string, std::string> params;
    params["align_corners"] = std::to_string(int(align_corner));
    params["factor"] = std::to_string(1);
    tensor_test_params input_dims  = {1, test_params.c, test_params.ih, test_params.iw};
    tensor_test_params output_dims = {1, test_params.c, test_params.oh, test_params.ow};

    SetInputTensor(input_dims);
    SetOutputTensor(output_dims);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Interp").params(params), NetworkInitParams().layoutPreference(layoutPreference)));
    ASSERT_NO_FATAL_FAILURE(SetFirstInputToRange(-0.9f, 0.9f));

    auto inputBlob = _inputMap.begin()->second;

    ASSERT_TRUE(Infer());
    auto outputBlob = _outputMap.begin()->second;

    ref_interp(inputBlob, _refBlob, align_corner);

    float maxerr = 0.07f;

    CompareCommonAbsolute(outputBlob, _refBlob, maxerr);
}
