// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

struct offset_test_params {
    size_t order0;
    size_t order1;
    size_t order2;
    size_t order3;
};

PRETTY_PARAM(Offsets, offset_test_params);

// Show contents of offset test param by not hexadecimal but integer
static inline void PrintTo(const offset_test_params& param, ::std::ostream* os)
{
    *os << "{ " << param.order0 << ", " << param.order1 << ", " << param.order2 << ", " << param.order3 << "}";
}
typedef std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector> PermuteParams;

class myriadLayersPermuteTests_smoke: public myriadLayersTests_nightly, /*input tensor, order */
                                        public testing::WithParamInterface<PermuteParams> {
};

static void genRefData(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->getTensorDesc().getLayout();
    SizeVector dims = blob->getTensorDesc().getDims();

    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];
        float counter = 0.f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.25f;
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}

TEST_P(myriadLayersPermuteTests_smoke, Permute) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<PermuteParams>::GetParam();
    auto input_tensor = std::get<0>(p);
    auto order =        std::get<1>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    if (input_tensor.size()) {
        gen_dims(output_tensor, input_tensor.size(), input_tensor[order[3]],
                                                     input_tensor[order[2]], 
                                                     input_tensor[order[1]], 
                                                     input_tensor[order[0]]);
    }
    std::string orderStr;
    for (int i = 0; i < order.size() - 1; ++i) {
        orderStr += std::to_string(order[i]);
        orderStr += ",";
    }
    if (!order.empty()) {
        orderStr += std::to_string(order.back());
    }
    std::map<std::string, std::string> layer_params = {
              {"order", orderStr}
    };
    _genDataCallback = genRefData;
    _testNet.addLayer(LayerInitParams("Permute")
             .params(layer_params)
             .in({input_tensor})
             .out({output_tensor}),
             ref_permute_wrap);
    ASSERT_TRUE(generateNetAndInfer(NetworkInitParams().useHWOpt( CheckMyriadX() )));
    CompareCommonAbsolute(_outputMap.begin()->second, getReferenceOutput(), 0.0f);
}

static const std::vector<InferenceEngine::SizeVector> s_inTensors = {
    {1, 36, 19, 19},
    {1, 2, 7, 8},
    {1, 196, 12, 2}
};

static const std::vector<InferenceEngine::SizeVector> s_permuteTensors = {
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1}
};
