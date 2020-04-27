// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

typedef myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, int32_t>> myriadLayersTestsFlatten_nightly;

static void ref_flatten(const InferenceEngine::Blob::Ptr src,
                        InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);
    get_dims(dst, OW, OH, OC);

    ASSERT_EQ(IW * IH *IC, OW * OH * OC);

    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();

    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    size_t sz = IW * IH *IC;
    std::vector<uint16_t> temp(sz);
    uint16_t* pTmp = temp.data();
    ASSERT_NE(pTmp, nullptr);
    //HWC->CHW
    for (int32_t ic = 0; ic < IC; ++ic) {
        for (int32_t ih = 0; ih < IH; ++ih) {
            for (int32_t iw = 0; iw < IW; ++iw) {
                int32_t iidx = iw + IW * ( ih  + ic * IH );
                int32_t oodx = ic + IC * ( iw  + ih * IW );
                temp[iidx] = src_data[oodx];
            }
        }
    }
    //CHW->HWC
    for (int32_t ow = 0; ow < OW; ++ow) {
        for (int32_t oh = 0; oh < OH; ++oh) {
            for (int32_t oc = 0; oc < OC; ++oc) {
                int32_t iidx = ow + OW * ( oh  + oc * OH );
                int32_t oodx = oc + OC * ( ow  + oh * OW );
                dst_data[oodx] = temp[iidx];
            }
        }
    }
}

TEST_P(myriadLayersTestsFlatten_nightly, Flatten) {
    auto input = std::get<0>(GetParam());
    int32_t axis_val = std::get<1>(GetParam());
    IN_OUT_desc input_tensor;
    IN_OUT_desc output_tensor;
    input_tensor.push_back(input);
    SetInputTensors(input_tensor);
    SetInputReshape();
    InferenceEngine::SizeVector out_dims;
    if (input.size() < 4) {
        axis_val -= 1;
        axis_val %= input.size();
    }
    if (input.size() == 4) {
        ASSERT_EQ(input[0], 1);
    }
    out_dims.push_back(1);
    switch (axis_val) {
        case 0:
            ASSERT_NE(input.size(), 4);
            {
                int32_t count = 1;
                for ( auto val : input)
                    count *= val;
                out_dims.push_back(count);
            }
            break;
        case 1:
            {
                if (input.size() == 4) {
                    int32_t count = 1;
                    for ( auto val : input)
                        count *= val;
                    out_dims.push_back(count);
                }else if (input.size() == 3) {
                    out_dims.push_back(input[0]);
                    out_dims.push_back(input[1] *input[2]);
                }else if (input.size() == 2) {
                    out_dims = input;
                }
            }
            break;
        case 2:
            {
                ASSERT_NE(input.size(), 2);
                if (input.size() == 3) {
                    out_dims = input;
                }else if (input.size() == 4) {
                    out_dims.push_back(input[1]);
                    out_dims.push_back(input[2] *input[3]);
                }
            }
            break;
        case 3:
            ASSERT_EQ(input.size(), 4);
            out_dims = input;
            break;
        default:
            FAIL() << "Unsupported axis value";
    }
    output_tensor.push_back(out_dims);
    SetOutputTensors(output_tensor);
    std::map<std::string, std::string> params;
    params["axis"] = std::to_string(axis_val);
    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("Flatten").params(params)));
    ASSERT_TRUE(Infer());
    ref_flatten(_inputMap.begin()->second, _refBlob);
    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<InferenceEngine::SizeVector> s_flattenTensors = {
    {{1, 4, 8, 16}},

    // FIXME: the test is written for [N]HWC layout, but InferenceEngine doesn't have 3D HWC layout.
//    {{4, 16, 32}},

    {{64, 32}},
};

static std::vector<int32_t> s_flattenAxis = {
    1, 2, 3
};
